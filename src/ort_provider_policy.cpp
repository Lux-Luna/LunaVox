#include "ort_provider_policy.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace qwen3_tts {

namespace {

std::string join_csv(const std::vector<std::string> & items) {
    if (items.empty()) {
        return "";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << items[i];
    }
    return oss.str();
}

bool append_cpu_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider_CPU(/*use_arena=*/1);
        return true;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to append CPUExecutionProvider: ") + e.what();
        return false;
    }
}

bool append_cuda_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        return true;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to append CUDAExecutionProvider: ") + e.what();
        return false;
    }
}

bool append_coreml_provider(Ort::SessionOptions & opts, std::string & error_msg) {
    try {
        opts.AppendExecutionProvider(
            "CoreMLExecutionProvider",
            std::unordered_map<std::string, std::string>{});
        return true;
    } catch (const std::exception & e) {
        error_msg = std::string("Failed to append CoreMLExecutionProvider: ") + e.what();
        return false;
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

    std::vector<std::string> enabled;
    std::vector<std::string> skipped;

    if (role == ort_session_role::cpu_only) {
        if (!append_cpu_provider(opts, error_msg)) {
            return false;
        }
        policy_summary = "enabled=cpu";
        return true;
    }

    std::string provider_error;
    if (append_cuda_provider(opts, provider_error)) {
        enabled.push_back("cuda");
    } else {
        skipped.push_back("cuda(" + provider_error + ")");
    }

    provider_error.clear();
    if (append_coreml_provider(opts, provider_error)) {
        enabled.push_back("coreml");
    } else {
        skipped.push_back("coreml(" + provider_error + ")");
    }

    std::string cpu_error;
    if (!append_cpu_provider(opts, cpu_error)) {
        error_msg = "Decoder provider policy failed and CPU fallback was unavailable: " + cpu_error;
        return false;
    }
    enabled.push_back("cpu");

    policy_summary = "enabled=" + join_csv(enabled);
    if (!skipped.empty()) {
        policy_summary += "; skipped=" + join_csv(skipped);
    }
    return true;
}

} // namespace qwen3_tts
