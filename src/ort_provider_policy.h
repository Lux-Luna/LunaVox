#pragma once

#include <string>

// Forward declare Ort types to avoid header issues in the interface if possible,
// but for SessionOptions we really need the header.
#include <onnxruntime_cxx_api.h>

namespace qwen3_tts {

enum class ort_session_role {
    cpu_only,
    decoder,
};

// Apply provider policy at session creation time.
// Returns false only when a mandatory provider setup step fails.
bool apply_ort_provider_policy(
    Ort::SessionOptions & opts,
    Ort::Env & env,
    ort_session_role role,
    std::string & error_msg,
    std::string & policy_summary);

} // namespace qwen3_tts
