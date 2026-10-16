def to_amp_phase_torch(iq_data):
    """
    使用 PyTorch 操作将 I/Q 数据转换为幅度和相位。
    输入 shape: (N, 2, 1024)
    输出 shape: (N, 1024, 2)
    """
    # 确保输入是 PyTorch 张量
    if not isinstance(iq_data, torch.Tensor):
        iq_data = torch.from_numpy(iq_data).float()

    # 创建复数张量
    complex_data = torch.complex(iq_data[:, 0, :], iq_data[:, 1, :])  # Shape: (N, 1024)

    # 计算幅度和相位
    amp = torch.abs(complex_data)
    phase = torch.angle(complex_data) / torch.pi  # PyTorch angle 直接返回弧度, 除以pi归一化

    # 堆叠并调整维度
    # amp 和 phase 的 shape 都是 (N, 1024)
    # 使用 unsqueeze(-1) 增加一个新维度 -> (N, 1024, 1)
    # 然后在最后一个维度上拼接
    amp_phase_data = torch.cat((amp.unsqueeze(-1), phase.unsqueeze(-1)), dim=-1)  # Shape: (N, 1024, 2)

    return amp_phase_data