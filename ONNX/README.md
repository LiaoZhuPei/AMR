下面分两部分给你:

---

# 一、相比原始单模型程序，多级识别版本的主要改动点

| 序号 | 改动点 | 说明 |
|---|---|---|
| 1 | **移除 `labels.txt` 外部文件依赖** | 原来用 `readClassNames()` 从文件读标签，改为三组 `std::vector<std::string>` 直接硬编码在代码里（`labels_stage1`/`labels_gmsk`/`labels_qam`），对应三个不同的 ONNX 模型。 |
| 2 | **预处理与推理解耦** | 原来 `run_iq_inference()` 把"功率归一化 + 交错排列 + 跑模型 + 打印"全部耦合在一个函数里，只能对应一个模型。拆分成 `preprocess_iq()`（只做一次数据预处理）+ `run_model_inference()`（通用推理，可反复喂给不同 session），避免二级推理时重复做归一化计算，逻辑也更清晰。 |
| 3 | **推理结果结构化** | 新增 `InferenceResult` 结构体（`index`/`label`/`confidence`/`probs`），替代原来直接在函数里打印、返回裸 `int index` 的方式，方便在多级之间传递和比较结果。 |
| 4 | **三模型并行加载** | 原来只加载 1 个 `Ort::Session`，现在同时加载并常驻 3 个 session（一级17类、二级GMSK组3类、二级QAM组5类），分别维护各自的 input/output 节点名。 |
| 5 | **路由判断逻辑** | 新增 `needGmskSecondStage()` / `needQamSecondStage()` 两个判断函数：一级结果若为 `2ASK/BPSK/GMSK` → 进 GMSK 二级模型；若标签中含 `"QAM"` → 进 QAM 二级模型；否则一级结果直接作为最终结果，不做二级推理。 |
| 6 | **NaN/Inf 防御** | 针对排查中发现的 ONNX `atan2` 在 `(0,0)` 点退化为 `0/0` 导致 NaN 的问题（已在模型侧用 `safe_atan2` 修复根因），曾在 C++ 侧加过 `hasInvalidValue()` 检测 + 兜底回退逻辑（二级结果异常时保留一级结果），避免 NaN 被 `std::max_element` 静默误判成 index 0 对应的类别。 |
| 7 | **统计方式调整** | 原来用 `vector<int> prediction_counts(labels.size())` 按一级标签的 index 统计，现在改为 `std::map<std::string, int> final_prediction_counts` 按**最终标签字符串**统计，因为最终结果可能来自三个不同模型、不同的 index 空间，用字符串 key 更安全、不会错位。 |
| 8 | **打印内容增加层级信息** | 每个样本会依次打印"一级推理结果"→（若触发）"二级推理结果"→"最终结果（含是否经过二级推理的标注）"，而不是原来只打印一次最终预测。 |
| 9 | **测试数据目录变更** | `csv_folder_path` 从原来的单模型测试目录改为 `./data_GMSK1`（按你最新需求调整）。 |

---

# 二、精简通用版本（去除冗余检查，提升可扩展性）

设计思路：把"多级级联"抽象成一个**通用的 Stage（阶段）+ Rule（路由规则）**框架，而不是写死"GMSK组/QAM组"这种专用分支判断。以后如果要加第三级、第四级，或者改路由条件，只需要改配置数据，不需要改控制流代码。

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <chrono>
#include <map>
#ifdef _WIN32
#include <Windows.h>
#endif
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

// ===================== 基础工具 =====================

std::wstring ToWideString(const std::string& str) {
    if (str.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &w[0], n);
    return w;
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    float mx = *std::max_element(x.begin(), x.end());
    float sum = 0.f;
    for (size_t i = 0; i < x.size(); ++i) { y[i] = std::exp(x[i] - mx); sum += y[i]; }
    for (auto& v : y) v /= sum;
    return y;
}

bool loadCsvIQ(const std::string& path, std::vector<float>& I, std::vector<float>& Q, int expected_len = 1024) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string li, lq;
    if (!std::getline(file, li) || !std::getline(file, lq)) return false;

    auto parseRow = [](const std::string& line, std::vector<float>& out) {
        std::stringstream ss(line);
        std::string v;
        while (std::getline(ss, v, ',')) if (!v.empty()) out.push_back(std::stof(v));
    };
    I.clear(); Q.clear();
    parseRow(li, I);
    parseRow(lq, Q);
    return (int)I.size() == expected_len && (int)Q.size() == expected_len;
}

std::vector<float> preprocess(const std::vector<float>& I, const std::vector<float>& Q) {
    int L = (int)I.size();
    float power = 0.f;
    for (int i = 0; i < L; ++i) power += I[i] * I[i] + Q[i] * Q[i];
    power /= L;
    float scale = (power < 1e-10f) ? 1.0f : 1.0f / std::sqrt(power);

    std::vector<float> out(L * 2);
    for (int i = 0; i < L; ++i) {
        out[i * 2]     = I[i] * scale;
        out[i * 2 + 1] = Q[i] * scale;
    }
    return out;
}

// ===================== 通用模型阶段 =====================

struct ModelStage {
    std::string name;
    Ort::Session session{ nullptr };
    std::string input_name;
    std::string output_name;
    std::vector<std::string> labels;

    ModelStage(Ort::Env& env, const std::string& model_path,
               const std::string& stage_name, std::vector<std::string> cls)
        : name(stage_name), labels(std::move(cls))
    {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        OrtSessionOptionsAppendExecutionProvider_CPU(opts, 0);
        session = Ort::Session(env, ToWideString(model_path).c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        input_name = session.GetInputName(0, alloc);
        output_name = session.GetOutputName(0, alloc);
    }

    // 对一份预处理好的 (1,1024,2) 数据做推理，返回 <label, confidence, probs>
    std::tuple<std::string, float, std::vector<float>> infer(std::vector<float>& data, int seq_len = 1024) {
        std::array<int64_t, 3> shape{ 1, seq_len, 2 };
        auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());

        const char* in_names[] = { input_name.c_str() };
        const char* out_names[] = { output_name.c_str() };
        auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

        const float* raw = outputs[0].GetTensorMutableData<float>();
        int n = (int)outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
        auto probs = softmax(std::vector<float>(raw, raw + n));
        int idx = (int)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        return { labels[idx], probs[idx] * 100.f, probs };
    }
};

// ===================== 通用路由规则 =====================

struct RouteRule {
    std::function<bool(const std::string&)> condition;  // 上一级标签满足什么条件
    int next_stage;                                       // 触发哪个后续stage（下标）
};

// ===================== 级联推理引擎 =====================

class CascadeClassifier {
public:
    CascadeClassifier(Ort::Env& env) : env_(env) {}

    int addStage(const std::string& model_path, const std::string& name, std::vector<std::string> labels) {
        stages_.emplace_back(env_, model_path, name, std::move(labels));
        return (int)stages_.size() - 1;
    }

    void addRule(int from_stage, std::function<bool(const std::string&)> cond, int to_stage) {
        rules_[from_stage].push_back({ cond, to_stage });
    }

    // 从 stage 0 开始级联推理，返回每一级的 (stage名, label, confidence)
    std::vector<std::tuple<std::string, std::string, float>> run(std::vector<float> data) {
        std::vector<std::tuple<std::string, std::string, float>> trace;
        int cur = 0;
        while (cur != -1) {
            auto [label, conf, probs] = stages_[cur].infer(data);
            trace.emplace_back(stages_[cur].name, label, conf);

            int next = -1;
            if (rules_.count(cur)) {
                for (auto& r : rules_[cur]) {
                    if (r.condition(label)) { next = r.next_stage; break; }
                }
            }
            cur = next;
        }
        return trace;
    }

private:
    Ort::Env& env_;
    std::vector<ModelStage> stages_;
    std::map<int, std::vector<RouteRule>> rules_;
};

// ===================== 主程序 =====================

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IQ-Inference");
    CascadeClassifier classifier(env);

    // --- 注册三级模型（以后加模型/改类别只改这里） ---
    int stage1 = classifier.addStage("./AMR_tcn_sionnAll_1.onnx", "Stage1", {
        "2ASK", "BPSK", "QPSK", "8PSK", "OQPSK", "Pi4DQPSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "16APSK", "32APSK", "2FSK", "4FSK", "MSK", "GMSK"
    });
    int stageGmsk = classifier.addStage("./AMR_tcn_sionnAll_gmsk.onnx", "GMSK-Stage2",
        { "2ASK", "BPSK", "GMSK" });
    int stageQam = classifier.addStage("./AMR_tcn_v2_SionnaAll_qam.onnx", "QAM-Stage2",
        { "16QAM", "32QAM", "64QAM", "128QAM", "256QAM" });

    // --- 注册路由规则：stage1结果满足什么条件 -> 进入哪个stage ---
    classifier.addRule(stage1, [](const std::string& l) {
        return l == "2ASK" || l == "BPSK" || l == "GMSK";
    }, stageGmsk);

    classifier.addRule(stage1, [](const std::string& l) {
        return l.find("QAM") != std::string::npos;
    }, stageQam);

    // --- 扫描测试文件 ---
    std::string folder = "./data_GMSK1";
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(folder))
        if (e.path().extension() == ".csv") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());

    std::map<std::string, int> final_counts;
    int total = 0;

    for (auto& path : files) {
        std::vector<float> I, Q;
        if (!loadCsvIQ(path, I, Q)) continue;

        auto data = preprocess(I, Q);
        auto trace = classifier.run(data);

        std::cout << "[" << std::filesystem::path(path).filename().string() << "] ";
        for (auto& [stage_name, label, conf] : trace)
            std::cout << stage_name << ":" << label << "(" << conf << "%) -> ";
        std::cout << "FINAL: " << std::get<1>(trace.back()) << std::endl;

        final_counts[std::get<1>(trace.back())]++;
        total++;
    }

    std::cout << "\n===== Summary (" << total << " samples) =====" << std::endl;
    for (auto& [label, cnt] : final_counts)
        std::cout << "  " << label << ": " << cnt << " (" << (cnt * 100.0 / total) << "%)" << std::endl;

    return 0;
}
```

## 这个精简版的通用性体现在哪里

1. **新增/替换模型不需要改控制流**：只需要调用 `classifier.addStage(path, name, labels)` 注册一个新模型，`addRule(...)` 配置触发条件，不需要写新的 `if-else` 分支或专用函数。
2. **支持任意级联深度**：`run()` 里是一个 `while` 循环，不是写死的"一级→二级"，如果以后二级结果还要触发三级模型，只需要再 `addRule(stageQam, ..., stage3)` 即可，代码结构不用动。
3. **路由条件用 `std::function` 表达**，可以是精确匹配、子串匹配、甚至结合置信度阈值（比如 `[](const string& l){ return l=="BPSK"; }` 可以换成更复杂的 lambda）。
4. **去掉了之前专门为排查 NaN 加的调试打印、原始 logits 打印、I/Q 前5点打印等**，因为根因已经在模型侧修复，不再需要这些防御性代码，保持主流程干净。
5. **`ModelStage::infer()` 是完全通用的推理函数**，不关心是几分类、不关心标签内容，可以被任意阶段复用。

### 需要你注意的点

- 这个精简版**去掉了 NaN 防御**（按你的要求"没有冗余检查"），前提是你已经在模型侧用 `safe_atan2` 修复了根因。如果以后遇到新的数值异常场景，可以在 `ModelStage::infer()` 的 `probs` 计算后视情况再加检测，不影响整体架构。
- `files.begin()`/文件数量限制（原来的 `max_files_to_process`）在这个精简版里省略了，如果需要限制数量，可以在扫描文件后加一行 `if (files.size() > N) files.resize(N);`。
- 异常处理（`try/catch`）也做了精简，如果 `session.Run()` 抛异常会直接中断程序，生产环境建议按需要补充 `try/catch`。
