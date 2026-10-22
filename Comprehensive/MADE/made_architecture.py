"""
MADE: Masked Denoising AutoEncoder Architecture for RF Fingerprinting
Paper: "Scalable Deep Learning for RF Fingerprinting: The MADE Architecture 
        for Robust Physical-Layer Device Identification"

复现思路：
  1. 预处理模块 → 2. 特征提取（IQ / 循环稳态）→ 3. MDAE核心（编码器+Transformer）
  → 4. 双输出头（分类 + 重建）→ 5. 联合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. 预处理模块 (PREPROCESSING MODULE)
#    Input : Raw I/Q Samples  [batch, 2, 256]
#    Output: Patches           [batch, 8, 2, 32]  (sequence of patches)
# =============================================================================

class PreprocessingModule(nn.Module):
    """
    对原始 I/Q 信号做归一化，再分段（Sequence Patching）。
    patch_size=32 → 256/32 = 8 patches
    """
    def __init__(self, signal_len: int = 256, patch_size: int = 32):
        super().__init__()
        assert signal_len % patch_size == 0, "signal_len must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = signal_len // patch_size  # 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, 256]  (2 = I + Q channels)
        returns: [B, 8, 2, 32]
        """
        # Signal Normalization: zero-mean, unit-variance per sample
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True) + 1e-8
        x = (x - mean) / std

        # Sequence Patching: split last dim into num_patches segments
        B, C, L = x.shape
        x = x.view(B, C, self.num_patches, self.patch_size)  # [B,2,8,32]
        x = x.permute(0, 2, 1, 3)                            # [B,8,2,32]
        return x


# =============================================================================
# 2a. IQ Feature Processor
#     Input : [B, 8, 2, 32]  → Patch Embedding → [B, 8, 128]
# =============================================================================

class IQFeatureProcessor(nn.Module):
    """
    Flatten each patch (2×32=64) and project to d_model=128.
    """
    def __init__(self, patch_channels: int = 2, patch_size: int = 32, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(patch_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 8, 2, 32] → [B, 8, 128]"""
        B, N, C, P = x.shape
        x = x.reshape(B, N, C * P)   # [B, 8, 64]
        return self.proj(x)           # [B, 8, 128]


# =============================================================================
# 2b. Cyclostationary Feature Processor (Cyclo Mode)
#     Input : [B, 8, 2, 32]
#     Output: [B, 8, 128]  via Feature Fusion & Projection
# =============================================================================

class CyclostationaryFeatureProcessor(nn.Module):
    """
    Two parallel branches:
      - Normalized Signal Branch
      - Raw Signal Branch
    Concatenate → Feature Fusion & Projection → [B, 8, 128]
    """
    def __init__(self, patch_channels: int = 2, patch_size: int = 32, d_model: int = 128):
        super().__init__()
        in_dim = patch_channels * patch_size  # 64

        self.norm_branch = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
        )
        self.raw_branch = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
        )
        # Fusion: concat two d_model features → d_model
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 8, 2, 32] → [B, 8, 128]"""
        B, N, C, P = x.shape
        flat = x.reshape(B, N, C * P)  # [B, 8, 64]

        # Normalized branch: additional layer-norm before projection
        norm_flat = F.layer_norm(flat, flat.shape[-1:])
        h_norm = self.norm_branch(norm_flat)   # [B, 8, 128]
        h_raw  = self.raw_branch(flat)         # [B, 8, 128]

        h = torch.cat([h_norm, h_raw], dim=-1)  # [B, 8, 256]
        return self.fusion(h)                    # [B, 8, 128]


# =============================================================================
# 3. MASKING MODULE
#    Randomly mask a subset of patches (set to zero).
# =============================================================================

class MaskingModule(nn.Module):
    """Random Patch Masking — mask_ratio fraction of patches zeroed out."""
    def __init__(self, mask_ratio: float = 0.30):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, d_model]
        returns: (masked_x, mask)  where mask is bool [B, N], True = masked
        """
        B, N, D = x.shape
        num_masked = max(1, int(N * self.mask_ratio))

        # Random indices to mask
        rand   = torch.rand(B, N, device=x.device)
        ids    = rand.argsort(dim=1)
        mask   = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids[:, :num_masked], True)

        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, mask


# =============================================================================
# 4. DENOISING MODULE
#    Add Gaussian noise to the (already masked) features.
# =============================================================================

class DenoisingModule(nn.Module):
    """Gaussian Noise Injection during training."""
    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


# =============================================================================
# 5. ENCODER (MLP-based)
#    [B, N, 128] → [B, N, 128]
#    Architecture: Linear(128→256) → BN → GELU → Dropout → Linear(256→128)
# =============================================================================

class MLPEncoder(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),   # Layer 1: 128 → 256
            nn.BatchNorm1d(hidden_dim),        # BatchNorm (applied over flattened)
            nn.GELU(),
            nn.Dropout(dropout),              # Dropout(0.1)
            nn.Linear(hidden_dim, d_model),   # Layer 2: 256 → 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 128]"""
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        out = self.net(x_flat)
        return out.reshape(B, N, D)          # [B, N, 128]


# =============================================================================
# 6. POSITIONAL ENCODING
#    Sinusoidal positional encoding added to encoder output.
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, N, D] + positional info"""
        return x + self.pe[:, :x.size(1), :]


# =============================================================================
# 7. TRANSFORMER MODULE
#    2-layer Transformer Encoder; 8-head attention; d_model=128
# =============================================================================

class TransformerModule(nn.Module):
    """
    2 × TransformerEncoderLayer:
      Multi-Head Self-Attention (8 heads) → FFN → LayerNorm → Residual
    Output: [B, N, 128]
    """
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 512, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,   # Post-LN (standard)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 128] → [B, N, 128]"""
        return self.transformer(x)


# =============================================================================
# 8. CLASSIFICATION HEAD
#    Attention Pooling → MLP-Classifier → Predicted Labels
#    Also outputs Device Embedding [B, 128] for cosine-similarity open-set detection
# =============================================================================

class AttentionPooling(nn.Module):
    """Attention-based pooling: learns a query to weight all N patch embeddings."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, D]"""
        weights = self.attn(x)               # [B, N, 1]
        weights = F.softmax(weights, dim=1)  # normalize over patches
        return (x * weights).sum(dim=1)      # [B, D]


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = 128, num_classes: int = 20):
        super().__init__()
        self.pool   = AttentionPooling(d_model)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, 128]
        returns: logits [B, num_classes], device_embedding [B, 128]
        """
        emb    = self.pool(x)    # [B, 128]  ← Device Embedding
        logits = self.mlp(emb)   # [B, num_classes]
        return logits, emb


# =============================================================================
# 9. RECONSTRUCTION HEAD (DECODER)
#    [B, N, 128] → Decode back to original patch features [B, N, 64]
#    Linear(128→256) → BN → GELU → Dropout → Linear(256→128) → Reshape → [B,2,256]
# =============================================================================

class ReconstructionHead(nn.Module):
    def __init__(self, d_model: int = 128, patch_channels: int = 2, patch_size: int = 32):
        super().__init__()
        out_dim = patch_channels * patch_size  # 64
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),   # Layer 1: 128 → 256
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),       # Layer 2: 256 → 128
            nn.GELU(),
            nn.Linear(128, out_dim),   # Project to patch output
        )
        self.patch_channels = patch_channels
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 128]
        returns: reconstructed signal [B, 2, 256]
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        out = self.decoder(x_flat)           # [B*N, 64]
        out = out.reshape(B, N, self.patch_channels, self.patch_size)  # [B,8,2,32]
        out = out.permute(0, 2, 1, 3).reshape(B, self.patch_channels, -1)  # [B,2,256]
        return out


# =============================================================================
# 10. COMPLETE MADE MODEL
# =============================================================================

class MADE(nn.Module):
    """
    Masked Denoising AutoEncoder (MADE) for RF Fingerprinting.

    Modes:
      mode='iq'    : IQ Feature Processor
      mode='cyclo' : Cyclostationary Feature Processor

    Dual output:
      - classification logits [B, num_classes]
      - reconstructed signal  [B, 2, 256]
      - device embedding      [B, 128]  (for open-set cosine similarity)
    """

    def __init__(
        self,
        signal_len:    int   = 256,
        patch_size:    int   = 32,
        d_model:       int   = 128,
        num_classes:   int   = 20,
        nhead:         int   = 8,
        num_tf_layers: int   = 2,
        mask_ratio:    float = 0.30,
        noise_std:     float = 0.10,
        mode:          str   = 'iq',       # 'iq' | 'cyclo'
    ):
        super().__init__()

        self.preprocess = PreprocessingModule(signal_len, patch_size)

        if mode == 'iq':
            self.feature_extractor = IQFeatureProcessor(2, patch_size, d_model)
        elif mode == 'cyclo':
            self.feature_extractor = CyclostationaryFeatureProcessor(2, patch_size, d_model)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'iq' or 'cyclo'.")

        self.masking    = MaskingModule(mask_ratio)
        self.denoising  = DenoisingModule(noise_std)
        self.encoder    = MLPEncoder(d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        self.transformer = TransformerModule(d_model, nhead, d_model * 4, num_layers=num_tf_layers)

        # Dual Output Pathway
        self.cls_head   = ClassificationHead(d_model, num_classes)
        self.recon_head = ReconstructionHead(d_model)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 2, 256]  — raw I/Q samples

        Returns dict:
          'logits'     : [B, num_classes]
          'recon'      : [B, 2, 256]
          'embedding'  : [B, 128]
          'mask'       : [B, N]  bool  (which patches were masked)
        """
        # --- Preprocessing ---
        patches = self.preprocess(x)        # [B, 8, 2, 32]

        # --- Feature Extraction ---
        feats = self.feature_extractor(patches)   # [B, 8, 128]

        # --- Masking & Denoising (only meaningful during pre-training) ---
        feats_masked, mask = self.masking(feats)  # [B, 8, 128], [B, 8]
        feats_noisy = self.denoising(feats_masked) # [B, 8, 128]

        # --- MLP Encoder ---
        encoded = self.encoder(feats_noisy)       # [B, 8, 128]

        # --- Positional Encoding ---
        encoded = self.pos_enc(encoded)            # [B, 8, 128]

        # --- Transformer ---
        tf_out = self.transformer(encoded)         # [B, 8, 128]

        # --- Dual Output Pathway ---
        logits, emb = self.cls_head(tf_out)        # [B, C], [B, 128]
        recon       = self.recon_head(tf_out)      # [B, 2, 256]

        return {
            'logits'    : logits,
            'recon'     : recon,
            'embedding' : emb,
            'mask'      : mask,
        }


# =============================================================================
# 11. LOSS FUNCTION (Combined Classification + Reconstruction)
# =============================================================================

class MADELoss(nn.Module):
    """
    L_total = L_cls (CrossEntropy) + λ * L_recon (MSE on masked patches only)
    """
    def __init__(self, lambda_recon: float = 1.0):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.ce_loss  = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: dict, labels: torch.Tensor, original_signal: torch.Tensor):
        """
        outputs        : dict returned by MADE.forward()
        labels         : [B] long tensor of true class indices
        original_signal: [B, 2, 256] — the clean original signal
        """
        # Classification loss
        l_cls   = self.ce_loss(outputs['logits'], labels)

        # Reconstruction loss (MSE between reconstructed and original)
        l_recon = self.mse_loss(outputs['recon'], original_signal)

        total = l_cls + self.lambda_recon * l_recon
        return total, l_cls, l_recon


# =============================================================================
# 12. DEVICE REGISTRY — Open-Set / Known-Unknown Detection
#     Cosine similarity between query embedding and stored device prototypes.
# =============================================================================

class DeviceRegistry:
    """
    Stores per-device prototype embeddings (mean of training embeddings).
    At inference, unknown devices are detected by low max cosine similarity.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold  = threshold
        self.prototypes = {}   # {device_id: embedding tensor [D]}

    @torch.no_grad()
    def register(self, device_id: str, embeddings: torch.Tensor):
        """embeddings: [N, D] — stack of embeddings for this device"""
        self.prototypes[device_id] = F.normalize(embeddings.mean(0, keepdim=True), dim=-1)

    @torch.no_grad()
    def query(self, embedding: torch.Tensor):
        """
        embedding: [D] or [1, D]
        Returns: (best_device_id or 'Unknown', similarity_score)
        """
        if not self.prototypes:
            return 'Unknown', 0.0

        q = F.normalize(embedding.unsqueeze(0) if embedding.dim() == 1 else embedding, dim=-1)
        best_id, best_sim = 'Unknown', -1.0

        for dev_id, proto in self.prototypes.items():
            sim = (q * proto).sum().item()
            if sim > best_sim:
                best_sim = sim
                best_id  = dev_id

        if best_sim < self.threshold:
            return 'Unknown', best_sim
        return best_id, best_sim


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Model ----------
    model = MADE(
        signal_len=256,
        patch_size=32,
        d_model=128,
        num_classes=20,
        nhead=8,
        num_tf_layers=2,
        mask_ratio=0.30,
        noise_std=0.10,
        mode='iq',
    ).to(device)

    print("=" * 60)
    print("MADE Architecture Summary")
    print("=" * 60)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total:,}")
    print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---------- Forward pass ----------
    batch_size = 4
    x = torch.randn(batch_size, 2, 256).to(device)
    labels = torch.randint(0, 20, (batch_size,)).to(device)

    model.train()
    out = model(x)

    print("\n  Output shapes:")
    print(f"    logits    : {out['logits'].shape}")     # [4, 20]
    print(f"    recon     : {out['recon'].shape}")       # [4, 2, 256]
    print(f"    embedding : {out['embedding'].shape}")   # [4, 128]
    print(f"    mask      : {out['mask'].shape}")        # [4, 8]

    # ---------- Loss ----------
    criterion = MADELoss(lambda_recon=1.0)
    total_loss, l_cls, l_recon = criterion(out, labels, x)
    print(f"\n  Loss — total: {total_loss.item():.4f} | cls: {l_cls.item():.4f} | recon: {l_recon.item():.4f}")

    # ---------- Open-set detection demo ----------
    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    registry = DeviceRegistry(threshold=0.85)
    registry.register('DeviceA', out_eval['embedding'])

    test_emb = out_eval['embedding'][0]
    dev_id, sim = registry.query(test_emb)
    print(f"\n  Open-set query → device: '{dev_id}', similarity: {sim:.4f}")

    print("\n  All checks passed ✓")
