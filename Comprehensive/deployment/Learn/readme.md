---

### 精简重构代码 (`iq_classifier.hpp` / `main.cpp`)

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <filesystem>
#include "onnxruntime_cxx_api.h"

// ===================================================================
// 1. 结构定义：定义模块的标准输出接口
// ===================================================================
struct InferenceResult {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    std::vector<float> probabilities;
};

// ===================================================================
// 2. 核心类定义：负责模型的加载、预处理、推理与后处理
// ===================================================================
class IQClassifier {
public:
    IQClassifier(const std::string& model_path, const std::vector<std::string>& labels)
        : labels_(labels), env_(ORT_LOGGING_LEVEL_WARNING, "IQClassifier") 
    {
        // 1. 初始化 SessionOptions
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 跨平台模型路径适配（Windows 下 path.c_str() 自动返回 wchar_t*，Linux 返回 char*）
        std::filesystem::path p(model_path);
        session_ = std::make_unique<Ort::Session>(env_, p.c_str(), session_options);

        // 2. 自动获取模型输入/输出节点名称（适配现代 ONNXRuntime API）
        Ort::AllocatorWithDefaultOptions allocator;
        auto in_name = session_->GetInputNameAllocated(0, allocator);
        input_names_str_ = { in_name.get() };
        auto out_name = session_->GetOutputNameAllocated(0, allocator);
        output_names_str_ = { out_name.get() };

        input_names_ = { input_names_str_[0].c_str() };
        output_names_ = { output_names_str_[0].c_str() };
    }

    // 主推理入口：支持其他系统直接传入 I/Q 原始数据向量
    InferenceResult predict(const std::vector<float>& i_data, const std::vector<float>& q_data) {
        if (i_data.size() != EXPECTED_SAMPLES || q_data.size() != EXPECTED_SAMPLES) {
            std::cerr << "[Error] Input IQ size mismatch! Expected " << EXPECTED_SAMPLES << std::endl;
            return {};
        }

        // 步骤 A: 数据预处理（功率归一化 + IQ交错打包）
        std::vector<float> input_tensor_values(EXPECTED_SAMPLES * 2);
        preprocess(i_data, q_data, input_tensor_values);

        // 步骤 B: 构建 ONNX Runtime 张量 (Shape: [1, 1024, 2])
        std::array<int64_t, 3> input_shape{ 1, EXPECTED_SAMPLES, 2 };
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, 
            input_tensor_values.data(), 
            input_tensor_values.size(), 
            input_shape.data(), 
            input_shape.size()
        );

        // 步骤 C: 执行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{ nullptr },
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), 1
        );

        // 步骤 D: 后处理与结果打包
        const float* logits = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t class_count = output_info.GetShape()[1];

        return postprocess(logits, class_count);
    }

private:
    static constexpr int EXPECTED_SAMPLES = 1024;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> labels_;

    // 输入输出名称内存管理
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // 预处理：功率归一化后排布成 [I0, Q0, I1, Q1, ...]
    void preprocess(const std::vector<float>& i_in, const std::vector<float>& q_in, std::vector<float>& out) {
        float power_sum = 0.0f;
        for (int i = 0; i < EXPECTED_SAMPLES; ++i) {
            power_sum += i_in[i] * i_in[i] + q_in[i] * q_in[i];
        }
        float power = power_sum / static_cast<float>(EXPECTED_SAMPLES);
        float scale = (power < 1e-10f) ? 1.0f : (1.0f / std::sqrt(power));

        for (int i = 0; i < EXPECTED_SAMPLES; ++i) {
            out[i * 2]     = i_in[i] * scale;
            out[i * 2 + 1] = q_in[i] * scale;
        }
    }

    // 后处理：Softmax + 取最大概率标签
    InferenceResult postprocess(const float* logits, size_t class_count) {
        InferenceResult res;
        res.probabilities.resize(class_count);

        // Softmax
        float max_val = *std::max_element(logits, logits + class_count);
        float sum = 0.0f;
        for (size_t i = 0; i < class_count; ++i) {
            res.probabilities[i] = std::exp(logits[i] - max_val);
            sum += res.probabilities[i];
        }
        for (size_t i = 0; i < class_count; ++i) {
            res.probabilities[i] /= sum;
        }

        // ArgMax
        auto max_it = std::max_element(res.probabilities.begin(), res.probabilities.end());
        res.class_id = static_cast<int>(std::distance(res.probabilities.begin(), max_it));
        res.confidence = *max_it;
        if (res.class_id < static_cast<int>(labels_.size())) {
            res.class_name = labels_[res.class_id];
        }

        return res;
    }
};

// ===================================================================
// 3. 示例调用演示（模拟外部系统使用）
// ===================================================================
int main() {
    // 标签表
    const std::vector<std::string> labels = {
        "BPSK", "QPSK", "OQPSK", "2FSK", "4FSK", "16QAM", "64QAM", "Pi4DQPSK", "GMSK"
    };

    // 1. 初始化引擎（仅在系统启动时载入一次）
    IQClassifier classifier("./AMR_tcn_2601_0819.onnx", labels);

    // 2. 模拟从雷达/SDR硬件或通信链路中收到一帧 1024 点的数据
    std::vector<float> mock_i_data(1024, 0.5f);
    std::vector<float> mock_q_data(1024, -0.3f);

    // 3. 调用推理（直接面向内存操作，极其轻量）
    InferenceResult result = classifier.predict(mock_i_data, mock_q_data);

    // 4. 业务消费结果
    std::cout << "预测类别: " << result.class_name 
              << " (ID: " << result.class_id << ")"
              << ", 置信度: " << result.confidence * 100.0f << "%" << std::endl;

    return 0;
}
```

---

### 架构设计与组织逻辑解析

这个精简模板展示了工业级 C++ 推理模块的四个核心标准：

1. **流水线分离（Preprocess $\to$ Infer $\to$ Postprocess）**
   * **`preprocess`**: 负责硬件/协议原始数据（I/Q数组）到张量内存布局（交错平铺、功率规整）的转换。
   * **`predict`（Infer）**: 仅负责将连续内存包装为 `Ort::Value`，驱动底层计算图。
   * **`postprocess`**: 负责数学归一化（Softmax）与业务解码（Argmax、置信度提取）。
2. **零文件系统耦合（In-Memory Operation）**
   * 删除了所有 CSV 读取、文件夹循环、终端格式化等逻辑。在真实部署场景中，数据通常来自网卡（Socket）、DMA 缓冲区或消息队列，以内存指针/向量方式传递是标准做法。
3. **跨平台与现代化（Modern C++ & ONNXRuntime）**
   * 使用 `std::filesystem::path` 抹平了 Windows (`wchar_t*`) 与 Linux (`char*`) 在 ONNX 模型路径入参上的差异，移除了平台特有的 Win32 API。
   * 采用 `session_->GetInputNameAllocated()`，兼容现代版本 ONNXRuntime 的内存生命周期管理。
4. **易于拓展**
   * **若需支持 GPU/TensorRT**：只需在构造函数中的 `session_options` 添加 `OrtSessionOptionsAppendExecutionProvider_CUDA`。
   * **若需封装为 C 接口/动态库（.so/.dll）**：只需将 `IQClassifier` 指针通过 `extern "C"` 导出为 `handle` 即可，十分适合被 Python、C# 或主工程调用。
