#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <iomanip>
#ifdef _WIN32
#include <Windows.h>
#endif

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"

// ===================================================================
// 辅助函数定义
// ===================================================================

std::wstring ToWideString(const std::string& str) {
#ifdef _WIN32
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
#endif
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    if (sum == 0.0f) return std::vector<float>(input.size(), 1.0f / input.size());
    for (size_t i = 0; i < output.size(); ++i) output[i] /= sum;
    return output;
}

void printProb(const std::vector<float>& probabilities, const std::vector<std::string>& labels) {
    std::cout << "  预测概率分布:" << std::endl;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::string label_name = (i < labels.size()) ? labels[i] : ("类别" + std::to_string(i));
        if (probabilities[i] > 0.01f) {
            std::cout << "    - " << label_name << ": "
                << probabilities[i] * 100.0f << "%" << std::endl;
        }
    }
}

bool load_iq_from_csv_2_rows(const std::string& file_path,
    std::vector<float>& i_data,
    std::vector<float>& q_data) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开 .csv 文件: " << file_path << std::endl;
        return false;
    }
    i_data.clear();
    q_data.clear();
    std::string i_line, q_line;
    if (!std::getline(file, i_line) || !std::getline(file, q_line)) {
        std::cerr << "错误: 文件 " << file_path << " 不足两行。" << std::endl;
        return false;
    }
    file.close();

    std::stringstream i_ss(i_line), q_ss(q_line);
    std::string val_str;
    while (std::getline(i_ss, val_str, ',')) {
        try { if (!val_str.empty()) i_data.push_back(std::stof(val_str)); }
        catch (const std::exception&) { return false; }
    }
    while (std::getline(q_ss, val_str, ',')) {
        try { if (!val_str.empty()) q_data.push_back(std::stof(val_str)); }
        catch (const std::exception&) { return false; }
    }

    const int expected_samples = 1024;
    if (i_data.size() != expected_samples || q_data.size() != expected_samples) {
        std::cerr << "错误: 文件 " << file_path
            << " I/Q 数据点数不为 " << expected_samples
            << " (I:" << i_data.size() << ", Q:" << q_data.size() << ")"
            << std::endl;
        return false;
    }
    return true;
}

// ===================================================================
// 功率归一化
// ===================================================================
void power_normalize(const std::vector<float>& i_data,
    const std::vector<float>& q_data,
    std::vector<float>& i_out,
    std::vector<float>& q_out)
{
    const int num_samples = static_cast<int>(i_data.size());
    i_out.resize(num_samples);
    q_out.resize(num_samples);

    float power_sum = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        power_sum += i_data[i] * i_data[i] + q_data[i] * q_data[i];
    }
    float power = power_sum / static_cast<float>(num_samples);

    if (power < 1e-10f) {
        i_out = i_data;
        q_out = q_data;
        return;
    }

    float scale = 1.0f / std::sqrt(power);
    for (int i = 0; i < num_samples; ++i) {
        i_out[i] = i_data[i] * scale;
        q_out[i] = q_data[i] * scale;
    }
}

// ===================================================================
// ①: 预处理函数，这样一份数据可以传给多个模型，不用重复算
// ===================================================================
void preprocess_iq(const std::vector<float>& i_data,
    const std::vector<float>& q_data,
    std::vector<float>& preprocessed_data)
{
    const int num_samples = 1024;
    const int num_features = 2;

    std::vector<float> i_normed, q_normed;
    power_normalize(i_data, q_data, i_normed, q_normed);

    preprocessed_data.resize(num_samples * num_features);
    for (int i = 0; i < num_samples; ++i) {
        preprocessed_data[i * 2] = i_normed[i];       // I路
        preprocessed_data[i * 2 + 1] = q_normed[i];   // Q路
    }
}

// ===================================================================
// ②: 单次模型推理的结果结构体 + 通用推理函数
// ===================================================================
struct InferenceResult {
    int index = -1;
    std::string label = "Unknown";
    float confidence = 0.0f;
    std::vector<float> probs;
};

InferenceResult run_model_inference(
    Ort::Session& session,
    const std::vector<std::string>& input_node_names,
    const std::vector<std::string>& output_node_names,
    const std::vector<std::string>& labels,
    std::vector<float>& preprocessed_data)
{
    InferenceResult result;

    std::array<int64_t, 3> input_shape{ 1, 1024, 2 };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info,
        preprocessed_data.data(),
        preprocessed_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session.Run(
            Ort::RunOptions{ nullptr },
            inputNames.data(), &input_tensor, 1,
            outNames.data(), outNames.size()
        );
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNXRuntime 推理错误: " << e.what() << std::endl;
        return result;
    }

    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    auto output_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int class_count = static_cast<int>(output_shape[1]);

    std::vector<float> probs = softmax(std::vector<float>(pdata, pdata + class_count));
    int max_index = static_cast<int>(
        std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
        );

    result.index = max_index;
    result.label = (max_index < static_cast<int>(labels.size())) ? labels[max_index] : "Unknown";
    result.confidence = probs[max_index] * 100.0f;
    result.probs = probs;

    return result;
}

// ===================================================================
// ③: 路由判断函数 —— 决定是否需要二次推理，以及用哪个模型
// ===================================================================
bool needGmskSecondStage(const std::string& label) {
    return (label == "2ASK" || label == "BPSK" || label == "GMSK");
}

bool needQamSecondStage(const std::string& label) {
    // 只要标签里包含 "QAM" 字样即可（16QAM/32QAM/64QAM/128QAM/256QAM）
    return (label.find("QAM") != std::string::npos);
}

// ===================================================================
// 主函数
// ===================================================================
int main() {
    // --- 1. 配置区 ---
    // *************************************************************************

    // ===== ④: 三个模型路径 =====
    const char* stage1_model_path = "./AMR_tcn_first.onnx";
    const char* gmsk_model_path = "./AMR_tcn_second_gmsk.onnx";
    const char* qam_model_path = "./AMR_tcn_second_qam.onnx";

    // ===== ⑤: 三组类别标签，不再读 labels.txt =====
    std::vector<std::string> labels_stage1 = {
        "2ASK",
        "BPSK", "QPSK", "8PSK", "OQPSK", "Pi4DQPSK",
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
        "16APSK", "32APSK",
        "2FSK", "4FSK", "MSK", "GMSK",
    };

    std::vector<std::string> labels_gmsk = {
        "2ASK", "BPSK", "GMSK"
    };

    std::vector<std::string> labels_qam = {
        "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"
    };

    // ===== ⑥: 测试目录 =====
    std::string csv_folder_path = "./data_128QAM";
    int max_files_to_process = 100;
    // *************************************************************************

    // --- 2. 扫描文件夹中的所有 .csv 文件 ---
    std::vector<std::string> all_csv_files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(csv_folder_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                all_csv_files.push_back(entry.path().string());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "错误: 无法访问文件夹 '" << csv_folder_path << "'. " << e.what() << std::endl;
        return -1;
    }

    if (all_csv_files.empty()) {
        std::cout << "在文件夹 '" << csv_folder_path << "' 中没有找到 .csv 文件。" << std::endl;
        return 0;
    }
    std::sort(all_csv_files.begin(), all_csv_files.end());
    std::cout << "找到 " << all_csv_files.size() << " 个 .csv 文件。" << std::endl;

    // --- 3. 截取文件列表 ---
    std::vector<std::string> files_to_process;
    if (static_cast<int>(all_csv_files.size()) > max_files_to_process) {
        files_to_process.assign(all_csv_files.begin(),
            all_csv_files.begin() + max_files_to_process);
        std::cout << "将只处理前 " << max_files_to_process << " 个文件。" << std::endl;
    }
    else {
        files_to_process = all_csv_files;
    }

    // --- 4. ONNXRuntime 初始化（三个模型分别加载）---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IQ-Inference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

    // --- 4.1 一级模型（17类）---
    std::wstring stage1_path_w = ToWideString(stage1_model_path);
    Ort::Session session_stage1(env, stage1_path_w.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator1;
    std::vector<std::string> input_names_stage1 = { session_stage1.GetInputName(0, allocator1) };
    std::vector<std::string> output_names_stage1 = { session_stage1.GetOutputName(0, allocator1) };
    std::cout << "[Stage1] 模型加载完成: " << stage1_model_path << std::endl;

    // --- 4.2 GMSK 组二级模型（3类）---
    std::wstring gmsk_path_w = ToWideString(gmsk_model_path);
    Ort::Session session_gmsk(env, gmsk_path_w.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator2;
    std::vector<std::string> input_names_gmsk = { session_gmsk.GetInputName(0, allocator2) };
    std::vector<std::string> output_names_gmsk = { session_gmsk.GetOutputName(0, allocator2) };
    std::cout << "[Stage2-GMSK] 模型加载完成: " << gmsk_model_path << std::endl;

    // --- 4.3 QAM 组二级模型（5类）---
    std::wstring qam_path_w = ToWideString(qam_model_path);
    Ort::Session session_qam(env, qam_path_w.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator3;
    std::vector<std::string> input_names_qam = { session_qam.GetInputName(0, allocator3) };
    std::vector<std::string> output_names_qam = { session_qam.GetOutputName(0, allocator3) };
    std::cout << "[Stage2-QAM] 模型加载完成: " << qam_model_path << std::endl;

    // --- 5. 初始化统计变量（按最终结果统计，二级推理覆盖一级结果）---
    std::map<std::string, int> final_prediction_counts;
    int successful_predictions = 0;

    // --- 6. 推理循环 ---
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t idx = 0; idx < files_to_process.size(); ++idx) {
        const auto& file_path = files_to_process[idx];
        std::cout << "\n-----------------------------------------------------" << std::endl;
        std::cout << "--- [" << (idx + 1) << "/" << files_to_process.size()
            << "] 文件: " << std::filesystem::path(file_path).filename().string()
            << " ---" << std::endl;

        std::vector<float> i_data, q_data;
        if (!load_iq_from_csv_2_rows(file_path, i_data, q_data)) {
            std::cerr << "跳过文件: " << file_path << std::endl;
            continue;
        }

        // ===== 预处理（只做一次，二级推理复用同一份数据）=====
        std::vector<float> preprocessed_data;
        preprocess_iq(i_data, q_data, preprocessed_data);

        // ===== 一级推理 =====
        InferenceResult result1 = run_model_inference(
            session_stage1, input_names_stage1, output_names_stage1,
            labels_stage1, preprocessed_data
        );

        std::cout << "[一级推理 - Stage1] 标签: " << result1.label
            << "  置信度: " << result1.confidence << "%" << std::endl;
        printProb(result1.probs, labels_stage1);

        // ===== 二级路由判断 =====
        InferenceResult final_result = result1;  // 默认用一级结果
        bool second_stage_triggered = false;
        std::string second_stage_name;

        if (needGmskSecondStage(result1.label)) {
            second_stage_triggered = true;
            second_stage_name = "GMSK组";
            InferenceResult result2 = run_model_inference(
                session_gmsk, input_names_gmsk, output_names_gmsk,
                labels_gmsk, preprocessed_data
            );
            std::cout << "[二级推理 - " << second_stage_name << "] 标签: " << result2.label
                << "  置信度: " << result2.confidence << "%" << std::endl;
            printProb(result2.probs, labels_gmsk);
            final_result = result2;
        }
        else if (needQamSecondStage(result1.label)) {
            second_stage_triggered = true;
            second_stage_name = "QAM组";
            InferenceResult result2 = run_model_inference(
                session_qam, input_names_qam, output_names_qam,
                labels_qam, preprocessed_data
            );
            std::cout << "[二级推理 - " << second_stage_name << "] 标签: " << result2.label
                << "  置信度: " << result2.confidence << "%" << std::endl;
            printProb(result2.probs, labels_qam);
            final_result = result2;
        }

        // ===== 打印最终结果 =====
        std::cout << "==> 最终结果: " << final_result.label
            << "  (置信度: " << final_result.confidence << "%"
            << (second_stage_triggered ? ", 经过二级推理)" : ", 一级直出)")
            << std::endl;

        final_prediction_counts[final_result.label]++;
        successful_predictions++;

        // ===== 调试：打印原始读入数据 =====
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "[CPP] I[0:5] = ";
        for (int i = 0; i < 5; i++) std::cout << i_data[i] << " ";
        std::cout << std::endl;

        std::cout << "[CPP] Q[0:5] = ";
        for (int i = 0; i < 5; i++) std::cout << q_data[i] << " ";
        std::cout << std::endl;

        double power_sum_d = 0.0;
        for (int i = 0; i < 1024; i++)
            power_sum_d += (double)i_data[i] * i_data[i] + (double)q_data[i] * q_data[i];
        double power_d = power_sum_d / 1024.0;
        std::cout << "[CPP] power = " << power_d << std::endl;

        std::cout << "[CPP] interleaved[0:10] = ";
        for (int i = 0; i < 10; i++) std::cout << preprocessed_data[i] << " ";
        std::cout << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // --- 7. 打印最终统计摘要 ---
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "              最终统计摘要（基于最终结果）" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "计划处理文件数: " << files_to_process.size() << std::endl;
    std::cout << "成功推理次数:   " << successful_predictions << std::endl;
    std::cout << "总耗时:         " << duration << " ms" << std::endl;
    if (successful_predictions > 0) {
        std::cout << "平均单次耗时:   "
            << (duration / successful_predictions) << " ms" << std::endl;
    }
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "各类别最终预测统计:" << std::endl;

    std::string max_label;
    int max_count = 0;
    for (const auto& kv : final_prediction_counts) {
        float pct = (successful_predictions > 0)
            ? (kv.second * 100.0f / successful_predictions) : 0.0f;
        std::cout << "  - " << kv.first << ": "
            << kv.second << " 次"
            << " (" << pct << "%)" << std::endl;
        if (kv.second > max_count) {
            max_count = kv.second;
            max_label = kv.first;
        }
    }

    if (successful_predictions > 0) {
        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << "最多预测类别: " << max_label
            << " (" << max_count << " 次, "
            << (max_count * 100.0f / successful_predictions) << "%)" << std::endl;
    }
    std::cout << "=====================================================" << std::endl;

    return 0;
}