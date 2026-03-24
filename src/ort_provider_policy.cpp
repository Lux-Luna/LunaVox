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
        // Generic approach if OrtCUDAProviderOptions is problematic
        opts.AppendExecutionProvider("CUDAExecutionProvider", {});
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

// Helper: Try to find metadata.json near the binary
static std::string find_metadata_json() {
    const char* p_list[] = {"metadata.json", "lib/metadata.json", "../lib/metadata.json"};
    for (const char* p : p_list) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
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
        success = false; detail = "Vulkan EP support not yet explicitly encoded";
    } else if (intent == "OpenVINOExecutionProvider") {
        success = false; detail = "OpenVINO EP support not yet explicitly encoded";
    }

    if (success) {
        policy_summary = intent;
    } else {
        if (meta_found && intent != "CPUExecutionProvider" && intent != "unknown") {
            LOG_WARN("Requested provider '%s' failed to load: %s. Falling back to CPU.", intent.c_str(), detail.c_str());
        }
        append_cpu_provider(opts, detail);
        policy_summary = "CPUExecutionProvider";
    }

    return true;
}

} // namespace qwen3_tts
