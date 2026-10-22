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

// ===== 改动 1：打印概率时同时显示类别名称 =====
void printProb(const float* pdata, int class_count, const std::vector<std::string>& labels) {
    std::vector<float> values(pdata, pdata + class_count);
    std::vector<float> probabilities = softmax(values);
    std::cout << "  预测概率分布:" << std::endl;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        std::string label_name = (i < labels.size()) ? labels[i] : ("类别" + std::to_string(i));
        // 只打印概率 > 1% 的类别，减少输出噪声
        if (probabilities[i] > 0.01f) {
            std::cout << "    - " << label_name << ": "
                << probabilities[i] * 100.0f << "%" << std::endl;
        }
    }
}

std::vector<std::string> readClassNames(const std::string& labels_txt_file) {
    std::vector<std::string> classNames;
    std::ifstream fp(labels_txt_file);
    if (!fp.is_open()) {
        std::cerr << "错误: 无法打开标签文件: " << labels_txt_file << std::endl;
        exit(-1);
    }
    std::string name;
    while (std::getline(fp, name)) {
        if (!name.empty()) classNames.push_back(name);
    }
    fp.close();
    return classNames;
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
//功率归一化
// ===================================================================
/**
 * @brief 对 IQ 数据进行功率归一化（与训练代码完全一致）
 *
 * Python 原始代码:
 *   power = np.mean(I**2 + Q**2)
 *   if power < 1e-10: return signal_data
 *   return signal_data / np.sqrt(power)
 *
 * @param i_data  I路数据（1024个点）
 * @param q_data  Q路数据（1024个点）
 * @param i_out   归一化后的I路数据
 * @param q_out   归一化后的Q路数据
 */
void power_normalize(const std::vector<float>& i_data,
    const std::vector<float>& q_data,
    std::vector<float>& i_out,
    std::vector<float>& q_out)
{
    const int num_samples = static_cast<int>(i_data.size());
    i_out.resize(num_samples);
    q_out.resize(num_samples);

    // 计算平均功率: power = mean(I^2 + Q^2)
    float power_sum = 0.0f;
    for (int i = 0; i < num_samples; ++i) {
        power_sum += i_data[i] * i_data[i] + q_data[i] * q_data[i];
    }
    float power = power_sum / static_cast<float>(num_samples);


    // 功率过小则不归一化（与 Python 中 power < 1e-10 一致）
    if (power < 1e-10f) {
        i_out = i_data;
        q_out = q_data;
        return;
    }

    // 除以 sqrt(power)
    float scale = 1.0f / std::sqrt(power);
    for (int i = 0; i < num_samples; ++i) {
        i_out[i] = i_data[i] * scale;
        q_out[i] = q_data[i] * scale;
    }
}

/**
 * @brief 对加载的 IQ 数据进行预处理和推理。
 *
 * 预处理流程（与 Python 推理代码完全对齐）:
 *   1. 功率归一化
 *   2. 交错排列为 [I0, Q0, I1, Q1, ...] → shape (1, 1024, 2)
 *
 * @return 预测的类别索引；如果失败则返回 -1。
 */
int run_iq_inference(
    Ort::Session& session,
    const std::vector<std::string>& input_node_names,
    const std::vector<std::string>& output_node_names,
    const std::vector<std::string>& labels,
    const std::vector<float>& i_data,
    const std::vector<float>& q_data)
{
    const int num_samples = 1024;
    const int num_features = 2;




    // ===== 功率归一化（替换原来的 min-max 归一化）=====
    std::vector<float> i_normed, q_normed;
    power_normalize(i_data, q_data, i_normed, q_normed);

    // ===== 交错排列 [I0,Q0, I1,Q1, ...] → shape (1, 1024, 2) =====
    std::vector<float> preprocessed_data(num_samples * num_features);
    for (int i = 0; i < num_samples; ++i) {
        preprocessed_data[i * 2] = i_normed[i];   // I路
        preprocessed_data[i * 2 + 1] = q_normed[i];   // Q路
    }

    // 构建输入张量：shape = (1, 1024, 2)
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
        return -1;
    }

    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    auto output_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int class_count = static_cast<int>(output_shape[1]);
    int max_index = static_cast<int>(
        std::distance(pdata, std::max_element(pdata, pdata + class_count))
        );

    // 计算置信度
    std::vector<float> probs = softmax(std::vector<float>(pdata, pdata + class_count));
    float confidence = probs[max_index] * 100.0f;

    std::cout << "预测结果:" << std::endl;
    std::cout << "  - 标签ID:   " << max_index << std::endl;
    std::cout << "  - 标签名称: " << labels[max_index] << std::endl;
    std::cout << "  - 置信度:   " << confidence << "%" << std::endl;
    printProb(pdata, class_count, labels);

    // 调试：打印原始读入数据
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "[CPP] I[0:5] = ";
    for (int i = 0; i < 5; i++) std::cout << i_data[i] << " ";
    std::cout << std::endl;

    std::cout << "[CPP] Q[0:5] = ";
    for (int i = 0; i < 5; i++) std::cout << q_data[i] << " ";
    std::cout << std::endl;

    // 调试：打印归一化后数据
    // (放在 power_normalize 调用之后)
    double power_sum_d = 0.0;
    for (int i = 0; i < 1024; i++)
        power_sum_d += (double)i_data[i] * i_data[i] + (double)q_data[i] * q_data[i];
    double power_d = power_sum_d / 1024.0;
    std::cout << "[CPP] power = " << power_d << std::endl;

    std::cout << "[CPP] i_norm[0:5] = ";
    for (int i = 0; i < 5; i++) std::cout << i_normed[i] << " ";
    std::cout << std::endl;

    std::cout << "[CPP] interleaved[0:10] = ";
    for (int i = 0; i < 10; i++) std::cout << preprocessed_data[i] << " ";
    std::cout << std::endl;


    return max_index;
}


// ===================================================================
// 主函数
// ===================================================================
int main() {
    // --- 1. 配置区 ---
    // *************************************************************************

    //ONNX 模型
    const char* onnx_path_char = "./AMR_tcn_2601_0819.onnx";

    // 类别标签
    // 可以选择从文件读取，也可以直接硬编码
    // 方式 A：硬编码
    //std::vector<std::string> labels = {
    //    "BPSK", "QPSK", "OQPSK", "2FSK", "4FSK",
    //    "16QAM", "64QAM", "Pi4DQPSK", "GMSK"
    //};
    // 方式 B：从文件读取
    std::string labels_txt_file = "./labels.txt";
    std::vector<std::string> labels = readClassNames(labels_txt_file);

    std::cout << "读取到 " << labels.size() << " 个标签:" << std::endl;
    for (size_t i = 0; i < labels.size(); i++)
        std::cout << "  [" << i << "] = '" << labels[i] << "'" << std::endl;

    std::string csv_folder_path = "./data_GMSK11";
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
    // 排序确保可复现
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

    // --- 4. ONNXRuntime 初始化 ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IQ-Inference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

    std::wstring model_path_w = ToWideString(onnx_path_char);
    Ort::Session session(env, model_path_w.c_str(), session_options);

    std::cout << "ONNXRuntime 初始化完成，模型: " << onnx_path_char << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_node_names = { session.GetInputName(0, allocator) };
    std::vector<std::string> output_node_names = { session.GetOutputName(0, allocator) };

    // 打印模型输入输出信息（调试用）
    std::cout << "模型输入节点: " << input_node_names[0] << std::endl;
    std::cout << "模型输出节点: " << output_node_names[0] << std::endl;

    auto input_shape_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "模型期望输入 shape: [";
    for (size_t i = 0; i < input_shape_info.size(); ++i) {
        std::cout << input_shape_info[i];
        if (i < input_shape_info.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // --- 5. 初始化统计变量 ---
    std::vector<int> prediction_counts(labels.size(), 0);
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
        if (load_iq_from_csv_2_rows(file_path, i_data, q_data)) {
            int predicted_index = run_iq_inference(
                session, input_node_names, output_node_names,
                labels, i_data, q_data
            );
            if (predicted_index >= 0 &&
                predicted_index < static_cast<int>(prediction_counts.size())) {
                prediction_counts[predicted_index]++;
                successful_predictions++;
            }
        }
        else {
            std::cerr << "跳过文件: " << file_path << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // --- 7. 打印最终统计摘要 ---
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "              最终统计摘要" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "计划处理文件数: " << files_to_process.size() << std::endl;
    std::cout << "成功推理次数:   " << successful_predictions << std::endl;
    std::cout << "总耗时:         " << duration << " ms" << std::endl;
    if (successful_predictions > 0) {
        std::cout << "平均单次耗时:   "
            << (duration / successful_predictions) << " ms" << std::endl;
    }
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "各类别预测统计:" << std::endl;

    // 找出预测最多的类别
    int max_count = 0;
    int max_count_idx = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        float pct = (successful_predictions > 0)
            ? (prediction_counts[i] * 100.0f / successful_predictions) : 0.0f;
        std::cout << "  - " << labels[i] << ": "
            << prediction_counts[i] << " 次"
            << " (" << pct << "%)" << std::endl;
        if (prediction_counts[i] > max_count) {
            max_count = prediction_counts[i];
            max_count_idx = static_cast<int>(i);
        }
    }

    if (successful_predictions > 0) {
        std::cout << "-----------------------------------------------------" << std::endl;
        std::cout << "最多预测类别: " << labels[max_count_idx]
            << " (" << max_count << " 次, "
                << (max_count * 100.0f / successful_predictions) << "%)" << std::endl;
    }
    std::cout << "=====================================================" << std::endl;

    return 0;
}