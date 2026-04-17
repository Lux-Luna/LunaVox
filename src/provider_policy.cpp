#include "provider_policy.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#include "json_utils.h"
#include "logger.h"

namespace lunavox {

namespace {

std::vector<std::string> load_build_capabilities() {
    std::vector<std::string> caps;
    const std::string metadata_path = find_metadata_json();
    if (metadata_path.empty()) {
        return caps;
    }
    std::ifstream f(metadata_path);
    if (!f.is_open()) {
        return caps;
    }
    const std::string content((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    const size_t onnx_start = content.find("\"onnx\"");
    if (onnx_start == std::string::npos) {
        return caps;
    }
    const std::string onnx_section = content.substr(onnx_start);

    // Prefer the new build_capabilities array; "provider" is the single-EP
    // form we still emit for GUI consumption, so fall back to it when no
    // array is present (older metadata files or hand-edited ones).
    if (json_extract_string_array(onnx_section, "build_capabilities", caps) &&
        !caps.empty()) {
        return caps;
    }
    std::string single;
    if (json_extract_string(onnx_section, "provider", single) && !single.empty()) {
        caps.push_back(single);
    }
    return caps;
}

bool has_cap(const std::vector<std::string> & caps, const std::string & name) {
    return std::find(caps.begin(), caps.end(), name) != caps.end();
}

ProviderCandidate make_cpu_candidate(bool decoder_role) {
    ProviderCandidate c;
    c.name = "CPUExecutionProvider";
    c.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    // Decoder graph is large; keep the allocator lean to contain peak RSS.
    c.disable_mem_arena = decoder_role;
    c.disable_mem_pattern = decoder_role;
    c.disable_spinning = decoder_role;
    c.apply = [](Ort::SessionOptions & opts, std::string & err) -> bool {
        try {
            opts.AppendExecutionProvider_CPU(/*use_arena=*/1);
            return true;
        } catch (const std::exception & e) {
            err = std::string("CPU EP error: ") + e.what();
            return false;
        }
    };
    return c;
}

ProviderCandidate make_cuda_candidate(bool low_mem) {
    ProviderCandidate c;
    c.name = "CUDAExecutionProvider";
    c.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    c.disable_mem_arena = true;
    c.disable_mem_pattern = true;
    c.disable_spinning = true;
    c.apply = [low_mem](Ort::SessionOptions & opts, std::string & err) -> bool {
        try {
            Ort::CUDAProviderOptions cuda_opts;
            // The decoder graph has many 1D convs; nc1d padding mode gives
            // cuDNN better kernel coverage on those.
            const char * max_workspace = low_mem ? "0" : "1";
            cuda_opts.Update({
                {"device_id", "0"},
                {"cudnn_conv_algo_search", "DEFAULT"},
                {"cudnn_conv_use_max_workspace", max_workspace},
                {"cudnn_conv1d_pad_to_nc1d", "1"},
            });
            opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
            return true;
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    };
    return c;
}

ProviderCandidate make_dml_candidate() {
    ProviderCandidate c;
    c.name = "DmlExecutionProvider";
    c.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    c.disable_mem_arena = true;
    c.disable_mem_pattern = true;
    c.disable_spinning = true;
    c.apply = [](Ort::SessionOptions & opts, std::string & err) -> bool {
        try {
            // disable_metacommands=1 bypasses fused DML metacommand kernels;
            // reduces workspace at a small decode latency cost. Env-gated.
            const char * no_meta = std::getenv("LUNAVOX_DML_NO_METACOMMANDS");
            std::unordered_map<std::string, std::string> dml_opts{{"device_id", "0"}};
            if (no_meta && no_meta[0] && no_meta[0] != '0') {
                dml_opts["disable_metacommands"] = "1";
            }
            opts.AppendExecutionProvider("DmlExecutionProvider", dml_opts);
            return true;
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    };
    return c;
}

ProviderCandidate make_rocm_candidate() {
    ProviderCandidate c;
    c.name = "ROCmExecutionProvider";
    c.opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    c.disable_mem_arena = true;
    c.disable_mem_pattern = true;
    c.disable_spinning = true;
    c.apply = [](Ort::SessionOptions & opts, std::string & err) -> bool {
        try {
            opts.AppendExecutionProvider("ROCmExecutionProvider", {});
            return true;
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    };
    return c;
}

ProviderCandidate make_coreml_candidate(std::string label,
                                        const char * model_format,
                                        const char * compute_units,
                                        GraphOptimizationLevel opt) {
    ProviderCandidate c;
    c.name = std::move(label);
    c.opt_level = opt;
    c.disable_mem_arena = true;
    c.disable_mem_pattern = true;
    c.disable_spinning = true;
    c.apply = [model_format, compute_units](Ort::SessionOptions & opts,
                                            std::string & err) -> bool {
        try {
            opts.AppendExecutionProvider("CoreMLExecutionProvider", {
                {"ModelFormat", model_format},
                {"MLComputeUnits", compute_units},
                {"RequireStaticInputShapes", "0"},
                {"EnableOnSubgraphs", "1"},
            });
            return true;
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    };
    return c;
}

} // namespace

std::vector<ProviderCandidate> build_provider_chain(ort_session_role role) {
    std::vector<ProviderCandidate> chain;

    if (role == ort_session_role::cpu_only) {
        chain.push_back(make_cpu_candidate(/*decoder_role=*/false));
        return chain;
    }

    // Decoder: try the platform's preferred accelerator(s) first, with a
    // per-platform graceful-degradation path, then universal CPU fallback.
    const auto caps = load_build_capabilities();

#if defined(__APPLE__)
    if (has_cap(caps, "CoreMLExecutionProvider")) {
        // Primary: MLProgram on CPU+GPU with full optimizations. This is the
        // fast path when the ORT CoreML partitioner handles our fp16 decoder
        // graph cleanly.
        chain.push_back(make_coreml_candidate(
            "CoreMLExecutionProvider(MLProgram,CPUAndGPU)",
            "MLProgram", "CPUAndGPU",
            GraphOptimizationLevel::ORT_ENABLE_ALL));
        // Fallback within CoreML: NeuralNetwork format uses a different
        // partitioner and often dodges the MLProgram "axis N not in valid
        // range" shape-propagation bug triggered by dynamo-exported
        // 4D KV-cache attention graphs on ORT 1.24+.
        chain.push_back(make_coreml_candidate(
            "CoreMLExecutionProvider(NeuralNetwork,CPUOnly)",
            "NeuralNetwork", "CPUOnly",
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED));
    }
#elif defined(_WIN32)
    if (has_cap(caps, "DmlExecutionProvider")) {
        chain.push_back(make_dml_candidate());
    } else if (has_cap(caps, "CUDAExecutionProvider")) {
        chain.push_back(make_cuda_candidate(/*low_mem=*/true));
    }
#else  // Linux / other POSIX
    if (has_cap(caps, "CUDAExecutionProvider")) {
        chain.push_back(make_cuda_candidate(/*low_mem=*/true));
    } else if (has_cap(caps, "ROCmExecutionProvider")) {
        chain.push_back(make_rocm_candidate());
    }
#endif

    chain.push_back(make_cpu_candidate(/*decoder_role=*/true));
    return chain;
}

} // namespace lunavox
