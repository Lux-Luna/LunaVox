#include "ort_provider_policy.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include "json_utils.h"
#include "logger.h"

namespace qwen3_tts {

namespace {

/**
 * Comprehensive Execution Provider (EP) registration helpers.
 */

bool append_cpu_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider_CPU(/*use_arena=*/1);
        return true;
    } catch (const std::exception & e) {
        error_msg = std::string("CPU EP error: ") + e.what();
        return false;
    }
}

bool append_cuda_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        // CUDA is not supported by the generic provider-name API in ORT 1.24.
        Ort::CUDAProviderOptions cuda_opts;
        // The decoder graph is convolution-heavy and includes 1D convs. Use the
        // nc1d padding mode recommended by ORT for better cuDNN coverage.
        cuda_opts.Update({
            {"device_id", "0"},
            {"cudnn_conv_algo_search", "DEFAULT"},
            {"cudnn_conv_use_max_workspace", "1"},
            {"cudnn_conv1d_pad_to_nc1d", "1"},
        });
        opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

bool append_dml_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        // Generic approach for DirectML
        opts.AppendExecutionProvider("DmlExecutionProvider", {{"device_id", "0"}});
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

bool append_rocm_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider("ROCmExecutionProvider", {});
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

bool append_coreml_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider("CoreMLExecutionProvider", {});
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

bool append_vulkan_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider("VulkanExecutionProvider", {});
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

bool append_openvino_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider("OpenVINOExecutionProvider", {});
        return true;
    } catch (const std::exception & e) {
        error_msg = e.what(); return false;
    }
}

} // namespace

bool apply_ort_provider_policy(
    Ort::SessionOptions & opts,
    Ort::Env & env,
    ort_session_role role,
    std::string & error_msg,
    std::string & policy_summary) {
    (void) env;
    error_msg.clear();
    policy_summary.clear();

    if (role == ort_session_role::cpu_only) {
        append_cpu_provider(opts, error_msg);
        policy_summary = "CPU (forced)";
        return true;
    }

    // 1. Try to load intent from metadata.json
    std::string intent = "CPUExecutionProvider";
    std::string metadata_path = find_metadata_json();
    bool meta_found = false;

    if (!metadata_path.empty()) {
        std::ifstream f(metadata_path);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            size_t onnx_start = content.find("\"onnx\"");
            if (onnx_start != std::string::npos) {
                std::string onnx_section = content.substr(onnx_start);
                if (qwen3_tts::json_extract_string(onnx_section, "provider", intent)) {
                    meta_found = true;
                }
            }
        }
    }

    bool success = false;
    std::string detail;

    // 2. Load requested provider
    if (intent == "CUDAExecutionProvider") {
        success = append_cuda_provider(opts, detail);
    } else if (intent == "DmlExecutionProvider") {
        success = append_dml_provider(opts, detail);
    } else if (intent == "ROCmExecutionProvider") {
        success = append_rocm_provider(opts, detail);
    } else if (intent == "CoreMLExecutionProvider") {
        success = append_coreml_provider(opts, detail);
    } else if (intent == "VulkanExecutionProvider") {
        success = append_vulkan_provider(opts, detail);
    } else if (intent == "OpenVINOExecutionProvider") {
        success = append_openvino_provider(opts, detail);
    }

    if (success) {
        policy_summary = intent;
    } else {
        if (meta_found && intent != "CPUExecutionProvider" && intent != "unknown") {
            std::string target = intent;
            size_t pos = target.find("ExecutionProvider");
            if (pos != std::string::npos) target = target.substr(0, pos);
            LOG_WARN("ONNX Runtime dependency (%s) is incomplete: %s. Falling back to CPU.", target.c_str(), detail.c_str());
        }
        append_cpu_provider(opts, detail);
        policy_summary = "CPUExecutionProvider";
    }

    return true;
}

} // namespace qwen3_tts
