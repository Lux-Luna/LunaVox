/**
 * LunaVox C++ Accelerator Plugin
 * 
 * Replaces the Python T2S autoregressive inference loop with a native C++ implementation
 * that calls ONNX Runtime C API directly, eliminating per-step Python interpreter overhead.
 * 
 * Build: pybind11 + MSVC + onnxruntime.lib
 */

#ifdef _WIN32
#include <windows.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <onnxruntime_c_api.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>

namespace py = pybind11;

// ============================================================================
// ORT C API Helper Macros
// ============================================================================

static const OrtApi* g_ort_api = nullptr;

#define ORT_CHECK(expr) \
    do { \
        OrtStatus* _status = (expr); \
        if (_status != nullptr) { \
            const char* _msg = g_ort_api->GetErrorMessage(_status); \
            std::string err_msg = std::string("ORT Error: ") + _msg; \
            g_ort_api->ReleaseStatus(_status); \
            throw std::runtime_error(err_msg); \
        } \
    } while (0)


// ============================================================================
// ORT Session Wrapper (RAII)
// ============================================================================

struct OrtSessionWrapper {
    OrtSession* session = nullptr;
    OrtSessionOptions* session_options = nullptr;
    
    // Cached metadata
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<ONNXTensorElementDataType> input_types;
    
    OrtSessionWrapper() = default;
    
    void load(OrtEnv* env, const std::wstring& model_path, 
              const std::vector<std::string>& providers) {
        ORT_CHECK(g_ort_api->CreateSessionOptions(&session_options));
        ORT_CHECK(g_ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));
        ORT_CHECK(g_ort_api->SetSessionLogSeverityLevel(session_options, 3));
        
        // Add CUDA if requested
        for (const auto& prov : providers) {
            if (prov == "CUDAExecutionProvider") {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                g_ort_api->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_opts);
                break;
            }
        }
        
        ORT_CHECK(g_ort_api->CreateSession(env, model_path.c_str(), session_options, &session));
        
        // Cache input/output names
        OrtAllocator* allocator;
        ORT_CHECK(g_ort_api->GetAllocatorWithDefaultOptions(&allocator));
        
        size_t num_inputs;
        ORT_CHECK(g_ort_api->SessionGetInputCount(session, &num_inputs));
        for (size_t i = 0; i < num_inputs; i++) {
            char* name;
            ORT_CHECK(g_ort_api->SessionGetInputName(session, i, allocator, &name));
            input_names.push_back(std::string(name));
            ORT_CHECK(g_ort_api->AllocatorFree(allocator, name));
            
            // Get input type
            OrtTypeInfo* type_info;
            ORT_CHECK(g_ort_api->SessionGetInputTypeInfo(session, i, &type_info));
            const OrtTensorTypeAndShapeInfo* tensor_info;
            ORT_CHECK(g_ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
            ONNXTensorElementDataType elem_type;
            ORT_CHECK(g_ort_api->GetTensorElementType(tensor_info, &elem_type));
            input_types.push_back(elem_type);
            g_ort_api->ReleaseTypeInfo(type_info);
        }
        
        size_t num_outputs;
        ORT_CHECK(g_ort_api->SessionGetOutputCount(session, &num_outputs));
        for (size_t i = 0; i < num_outputs; i++) {
            char* name;
            ORT_CHECK(g_ort_api->SessionGetOutputName(session, i, allocator, &name));
            output_names.push_back(std::string(name));
            ORT_CHECK(g_ort_api->AllocatorFree(allocator, name));
        }
    }
    
    bool has_input(const std::string& name) const {
        return std::find(input_names.begin(), input_names.end(), name) != input_names.end();
    }
    
    int find_input_index(const std::string& name) const {
        auto it = std::find(input_names.begin(), input_names.end(), name);
        if (it == input_names.end()) return -1;
        return static_cast<int>(it - input_names.begin());
    }
    
    int find_output_index(const std::string& name) const {
        auto it = std::find(output_names.begin(), output_names.end(), name);
        if (it == output_names.end()) return -1;
        return static_cast<int>(it - output_names.begin());
    }
    
    ~OrtSessionWrapper() {
        if (session) g_ort_api->ReleaseSession(session);
        if (session_options) g_ort_api->ReleaseSessionOptions(session_options);
    }
    
    // Non-copyable
    OrtSessionWrapper(const OrtSessionWrapper&) = delete;
    OrtSessionWrapper& operator=(const OrtSessionWrapper&) = delete;
};


// ============================================================================
// Helper: Create OrtValue from numpy array
// ============================================================================

static OrtValue* numpy_to_ort_value(py::array arr, OrtMemoryInfo* mem_info) {
    OrtValue* value = nullptr;
    
    // Determine element type
    ONNXTensorElementDataType ort_type;
    py::dtype dt = arr.dtype();
    
    if (dt.is(py::dtype::of<float>())) {
        ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (dt.is(py::dtype::of<double>())) {
        ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    } else if (dt.is(py::dtype::of<int64_t>())) {
        ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (dt.is(py::dtype::of<int32_t>())) {
        ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    } else if (dt.is(py::dtype("float16"))) {
        ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    } else {
        throw std::runtime_error("Unsupported numpy dtype for ORT conversion");
    }
    
    // Build shape
    std::vector<int64_t> shape(arr.ndim());
    for (int i = 0; i < arr.ndim(); i++) {
        shape[i] = arr.shape(i);
    }
    
    size_t data_size = static_cast<size_t>(arr.nbytes());
    
    ORT_CHECK(g_ort_api->CreateTensorWithDataAsOrtValue(
        mem_info,
        arr.mutable_data(),
        data_size,
        shape.data(),
        shape.size(),
        ort_type,
        &value
    ));
    
    return value;
}


// ============================================================================
// Helper: Cast numpy to match model expected type
// ============================================================================

static py::array cast_to_model_type(py::array arr, ONNXTensorElementDataType expected_type) {
    switch (expected_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            if (!arr.dtype().is(py::dtype::of<float>()))
                return arr.attr("astype")(py::dtype::of<float>());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            if (!arr.dtype().is(py::dtype("float16")))
                return arr.attr("astype")(py::dtype("float16"));
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            if (!arr.dtype().is(py::dtype::of<int64_t>()))
                return arr.attr("astype")(py::dtype::of<int64_t>());
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            if (!arr.dtype().is(py::dtype::of<int32_t>()))
                return arr.attr("astype")(py::dtype::of<int32_t>());
            break;
        default:
            break;
    }
    return arr;
}


// ============================================================================
// Core T2S Engine
// ============================================================================

class T2SEngine {
public:
    OrtEnv* env_ = nullptr;
    OrtSessionWrapper encoder_;
    OrtSessionWrapper first_stage_decoder_;
    OrtSessionWrapper stage_decoder_;
    OrtMemoryInfo* cpu_mem_info_ = nullptr;
    
    int max_steps_ = 500;
    bool loaded_ = false;
    
    T2SEngine() {
        g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!g_ort_api) {
            throw std::runtime_error("Failed to get ORT API");
        }
        ORT_CHECK(g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lunavox_acc", &env_));
        ORT_CHECK(g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem_info_));
    }
    
    ~T2SEngine() {
        if (cpu_mem_info_) g_ort_api->ReleaseMemoryInfo(cpu_mem_info_);
        if (env_) g_ort_api->ReleaseEnv(env_);
    }
    
    void load_models(const std::string& encoder_path,
                     const std::string& fsd_path,
                     const std::string& sd_path,
                     const std::vector<std::string>& providers) {
        // Convert to wide string for Windows
        auto to_wstring = [](const std::string& s) -> std::wstring {
            // Use MultiByteToWideChar on Windows
            int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
            std::wstring ws(len - 1, 0);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
            return ws;
        };
        
        encoder_.load(env_, to_wstring(encoder_path), providers);
        first_stage_decoder_.load(env_, to_wstring(fsd_path), providers);
        stage_decoder_.load(env_, to_wstring(sd_path), providers);
        loaded_ = true;
    }
    
    /**
     * Run the full T2S pipeline: Encoder → First Stage Decoder → Autoregressive Loop
     * Returns: int64 numpy array of semantic tokens
     */
    py::array_t<int64_t> run_t2s(
        py::array ref_seq,
        py::array ref_bert,
        py::array text_seq,
        py::array text_bert,
        py::array ssl_content
    ) {
        if (!loaded_) {
            throw std::runtime_error("Models not loaded. Call load_models() first.");
        }
        
        // Release GIL for the entire inference
        py::gil_scoped_release release_gil;
        
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // ===== 1. ENCODER =====
        std::vector<const char*> enc_in_names_c;
        std::vector<OrtValue*> enc_input_values;
        
        // Re-acquire GIL briefly for numpy operations
        {
            py::gil_scoped_acquire acquire_gil;
            
            // Map of input name -> numpy array
            struct NamedInput {
                std::string name;
                py::array arr;
            };
            std::vector<NamedInput> enc_inputs_list = {
                {"ref_seq", ref_seq},
                {"text_seq", text_seq},
                {"ref_bert", ref_bert},
                {"text_bert", text_bert},
                {"ssl_content", ssl_content},
            };
            
            for (auto& ni : enc_inputs_list) {
                int idx = encoder_.find_input_index(ni.name);
                if (idx >= 0) {
                    ni.arr = cast_to_model_type(ni.arr, encoder_.input_types[idx]);
                    // Ensure contiguous
                    ni.arr = py::array::ensure(ni.arr, py::array::c_style | py::array::forcecast);
                    enc_in_names_c.push_back(encoder_.input_names[idx].c_str());
                    enc_input_values.push_back(numpy_to_ort_value(ni.arr, cpu_mem_info_));
                }
            }
        }
        
        // Run encoder
        std::vector<const char*> enc_out_names_c;
        for (auto& n : encoder_.output_names) enc_out_names_c.push_back(n.c_str());
        
        std::vector<OrtValue*> enc_outputs(encoder_.output_names.size(), nullptr);
        ORT_CHECK(g_ort_api->Run(
            encoder_.session, nullptr,
            enc_in_names_c.data(), enc_input_values.data(), enc_input_values.size(),
            enc_out_names_c.data(), enc_out_names_c.size(),
            enc_outputs.data()
        ));
        
        // Release encoder input OrtValues
        for (auto* v : enc_input_values) g_ort_api->ReleaseValue(v);
        enc_input_values.clear();
        
        // ===== 2. FIRST STAGE DECODER =====
        // Build input map from encoder outputs
        std::vector<const char*> fsd_in_names_c;
        std::vector<OrtValue*> fsd_input_values;
        
        for (const auto& in_name : first_stage_decoder_.input_names) {
            int enc_out_idx = encoder_.find_output_index(in_name);
            if (enc_out_idx >= 0) {
                fsd_in_names_c.push_back(in_name.c_str());
                fsd_input_values.push_back(enc_outputs[enc_out_idx]);
            }
        }
        
        std::vector<const char*> fsd_out_names_c;
        for (auto& n : first_stage_decoder_.output_names) fsd_out_names_c.push_back(n.c_str());
        
        std::vector<OrtValue*> fsd_outputs(first_stage_decoder_.output_names.size(), nullptr);
        ORT_CHECK(g_ort_api->Run(
            first_stage_decoder_.session, nullptr,
            fsd_in_names_c.data(), fsd_input_values.data(), fsd_input_values.size(),
            fsd_out_names_c.data(), fsd_out_names_c.size(),
            fsd_outputs.data()
        ));
        
        // Don't release encoder outputs used as FSD inputs (they're shared ptrs)
        // But release the ones NOT used by FSD
        for (size_t i = 0; i < enc_outputs.size(); i++) {
            bool used = false;
            for (auto* v : fsd_input_values) {
                if (v == enc_outputs[i]) { used = true; break; }
            }
            if (!used && enc_outputs[i]) {
                g_ort_api->ReleaseValue(enc_outputs[i]);
                enc_outputs[i] = nullptr;
            }
        }
        
        // Parse FSD outputs
        auto fsd_get = [&](const std::string& name) -> OrtValue* {
            int idx = first_stage_decoder_.find_output_index(name);
            return (idx >= 0) ? fsd_outputs[idx] : nullptr;
        };
        
        OrtValue* d_y = fsd_get("y");
        OrtValue* d_y_emb = fsd_get("y_emb");
        OrtValue* d_x_example = fsd_get("x_example");
        OrtValue* d_k_agg = fsd_get("k");
        OrtValue* d_v_agg = fsd_get("v");
        
        // Collect per-layer KV caches from first stage (Variant B)
        std::vector<OrtValue*> fs_k_layers, fs_v_layers;
        for (size_t i = 0; i < first_stage_decoder_.output_names.size(); i++) {
            const auto& nm = first_stage_decoder_.output_names[i];
            if (nm.find("present_k_layer_") == 0) {
                fs_k_layers.push_back(fsd_outputs[i]);
            } else if (nm.find("present_v_layer_") == 0) {
                fs_v_layers.push_back(fsd_outputs[i]);
            }
        }
        
        // ===== 3. STAGE DECODER AUTOREGRESSIVE LOOP =====
        
        // Determine KV cache structure
        int n_past_k = 0, n_past_v = 0;
        for (const auto& nm : stage_decoder_.input_names) {
            if (nm.find("past_k_layer_") == 0) n_past_k++;
            if (nm.find("past_v_layer_") == 0) n_past_v++;
        }
        int n_layers = std::max(n_past_k, n_past_v);
        
        // Build initial KV cache map
        std::map<std::string, OrtValue*> past_kv;
        
        if (n_layers > 0 && !fs_k_layers.empty() && !fs_v_layers.empty()) {
            // Use per-layer caches from first stage directly
            for (int i = 0; i < std::min((int)fs_k_layers.size(), n_layers); i++) {
                past_kv["past_k_layer_" + std::to_string(i)] = fs_k_layers[i];
                past_kv["past_v_layer_" + std::to_string(i)] = fs_v_layers[i];
            }
        }
        // Note: Aggregated KV cache splitting (Variant A) skipped for now
        // as per-layer variant is the standard path
        
        // Extract first token from d_y
        int64_t first_token = 0;
        if (d_y) {
            int64_t* y_data;
            ORT_CHECK(g_ort_api->GetTensorMutableData(d_y, (void**)&y_data));
            first_token = y_data[0];
        }
        
        std::vector<int64_t> out_tokens;
        out_tokens.push_back(first_token);
        
        // Current state
        OrtValue* cur_iy = d_y;
        OrtValue* cur_iy_emb = d_y_emb;
        
        // Main autoregressive loop
        for (int step = 0; step < max_steps_; step++) {
            // Build inputs for this step
            std::vector<const char*> sd_in_names;
            std::vector<OrtValue*> sd_in_values;
            
            // iy, iy_emb
            if (stage_decoder_.has_input("iy") && cur_iy) {
                sd_in_names.push_back("iy");
                sd_in_values.push_back(cur_iy);
            }
            if (stage_decoder_.has_input("iy_emb") && cur_iy_emb) {
                sd_in_names.push_back("iy_emb");
                sd_in_values.push_back(cur_iy_emb);
            }
            
            // Static inputs
            if (stage_decoder_.has_input("ix_example") && d_x_example) {
                sd_in_names.push_back("ix_example");
                sd_in_values.push_back(d_x_example);
            }
            if (stage_decoder_.has_input("ik") && d_k_agg) {
                sd_in_names.push_back("ik");
                sd_in_values.push_back(d_k_agg);
            }
            if (stage_decoder_.has_input("iv") && d_v_agg) {
                sd_in_names.push_back("iv");
                sd_in_values.push_back(d_v_agg);
            }
            
            // Per-layer KV caches
            for (auto& [name, val] : past_kv) {
                if (stage_decoder_.has_input(name) && val) {
                    sd_in_names.push_back(name.c_str());
                    sd_in_values.push_back(val);
                }
            }
            
            // Prepare outputs
            std::vector<const char*> sd_out_names;
            for (auto& n : stage_decoder_.output_names) sd_out_names.push_back(n.c_str());
            
            std::vector<OrtValue*> sd_outputs(stage_decoder_.output_names.size(), nullptr);
            
            ORT_CHECK(g_ort_api->Run(
                stage_decoder_.session, nullptr,
                sd_in_names.data(), sd_in_values.data(), sd_in_values.size(),
                sd_out_names.data(), sd_out_names.size(),
                sd_outputs.data()
            ));
            
            // Parse outputs
            auto sd_get = [&](const std::string& name) -> OrtValue* {
                int idx = stage_decoder_.find_output_index(name);
                return (idx >= 0) ? sd_outputs[idx] : nullptr;
            };
            
            // Get samples
            OrtValue* d_samples = sd_get("samples");
            int64_t token_val = -1;
            
            if (d_samples) {
                int64_t* samples_data;
                ORT_CHECK(g_ort_api->GetTensorMutableData(d_samples, (void**)&samples_data));
                token_val = samples_data[0];
            } else {
                // Fallback to 'y' output
                OrtValue* d_y_out = sd_get("y");
                if (d_y_out) {
                    OrtTensorTypeAndShapeInfo* info;
                    ORT_CHECK(g_ort_api->GetTensorTypeAndShape(d_y_out, &info));
                    size_t elem_count;
                    ORT_CHECK(g_ort_api->GetTensorShapeElementCount(info, &elem_count));
                    g_ort_api->ReleaseTensorTypeAndShapeInfo(info);
                    
                    int64_t* y_data;
                    ORT_CHECK(g_ort_api->GetTensorMutableData(d_y_out, (void**)&y_data));
                    token_val = y_data[elem_count - 1];
                }
            }
            
            if (token_val < 0) break; // No valid output
            
            out_tokens.push_back(token_val);
            
            // Stop condition: EOS token
            if (token_val >= 1024) break;
            
            // Update state for next iteration
            // Release previous step's outputs that are being replaced
            // (except the first iteration where we use FSD outputs)
            
            // Update iy
            if (d_samples) {
                // Release old cur_iy if it's not from FSD
                if (step > 0 && cur_iy) g_ort_api->ReleaseValue(cur_iy);
                cur_iy = d_samples;
                sd_outputs[stage_decoder_.find_output_index("samples")] = nullptr; // Don't free
            }
            
            // Update y_emb
            OrtValue* new_y_emb = sd_get("y_emb");
            if (new_y_emb) {
                if (step > 0 && cur_iy_emb) g_ort_api->ReleaseValue(cur_iy_emb);
                cur_iy_emb = new_y_emb;
                sd_outputs[stage_decoder_.find_output_index("y_emb")] = nullptr;
            }
            
            // Update KV caches
            for (size_t i = 0; i < stage_decoder_.output_names.size(); i++) {
                const auto& nm = stage_decoder_.output_names[i];
                if (nm.find("present_k_layer_") == 0) {
                    std::string layer_str = nm.substr(strlen("present_k_layer_"));
                    std::string past_name = "past_k_layer_" + layer_str;
                    if (past_kv.count(past_name) && past_kv[past_name] && step > 0) {
                        g_ort_api->ReleaseValue(past_kv[past_name]);
                    }
                    past_kv[past_name] = sd_outputs[i];
                    sd_outputs[i] = nullptr;
                } else if (nm.find("present_v_layer_") == 0) {
                    std::string layer_str = nm.substr(strlen("present_v_layer_"));
                    std::string past_name = "past_v_layer_" + layer_str;
                    if (past_kv.count(past_name) && past_kv[past_name] && step > 0) {
                        g_ort_api->ReleaseValue(past_kv[past_name]);
                    }
                    past_kv[past_name] = sd_outputs[i];
                    sd_outputs[i] = nullptr;
                }
            }
            
            // Release remaining outputs that were not kept
            for (auto* v : sd_outputs) {
                if (v) g_ort_api->ReleaseValue(v);
            }
        }
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        
        // Cleanup: release remaining OrtValues
        // KV caches
        for (auto& [name, val] : past_kv) {
            if (val) g_ort_api->ReleaseValue(val);
        }
        // FSD outputs (only those not already released/transferred)
        for (size_t i = 0; i < fsd_outputs.size(); i++) {
            // Check if this output was transferred to past_kv or cur_iy/cur_iy_emb
            bool skip = false;
            if (fsd_outputs[i] == cur_iy || fsd_outputs[i] == cur_iy_emb) skip = true;
            for (auto& [_, v] : past_kv) {
                if (fsd_outputs[i] == v) { skip = true; break; }
            }
            if (!skip && fsd_outputs[i]) g_ort_api->ReleaseValue(fsd_outputs[i]);
        }
        // Remaining encoder outputs
        for (auto* v : enc_outputs) {
            if (v) g_ort_api->ReleaseValue(v);
        }
        
        // Re-acquire GIL and build result
        py::gil_scoped_acquire acquire_gil;
        
        // Build numpy result: shape (1, N)
        std::vector<int64_t> shape = {1, static_cast<int64_t>(out_tokens.size())};
        auto result = py::array_t<int64_t>(shape);
        auto buf = result.mutable_unchecked<2>();
        for (size_t i = 0; i < out_tokens.size(); i++) {
            buf(0, i) = out_tokens[i];
        }
        
        return result;
    }
    
    double get_load_time_ms() const { return load_time_ms_; }
    
    void set_max_steps(int steps) { max_steps_ = steps; }
    
private:
    double load_time_ms_ = 0.0;
};


// ============================================================================
// Pybind11 Module Definition
// ============================================================================

// Helper: auto-detect and add ORT DLL directory on Windows
static void setup_ort_dll_path() {
#ifdef _WIN32
    try {
        py::module_ ort_mod = py::module_::import("onnxruntime");
        py::object ort_file = ort_mod.attr("__file__");
        std::string ort_path = ort_file.cast<std::string>();
        // Get directory containing __init__.py
        size_t pos = ort_path.find_last_of("\\/");
        if (pos != std::string::npos) {
            std::string ort_dir = ort_path.substr(0, pos);
            std::string capi_dir = ort_dir + "\\capi";
            // Convert to wstring
            int len = MultiByteToWideChar(CP_UTF8, 0, capi_dir.c_str(), -1, nullptr, 0);
            std::wstring ws(len - 1, 0);
            MultiByteToWideChar(CP_UTF8, 0, capi_dir.c_str(), -1, &ws[0], len);
            AddDllDirectory(ws.c_str());
        }
    } catch (...) {
        // Silently fail - user may need to set paths manually
    }
#endif
}

PYBIND11_MODULE(lunavox_accelerator, m) {
    m.doc() = "LunaVox C++ Accelerator - Native T2S autoregressive inference loop";
    
    // Auto-detect ORT DLL path on import
    setup_ort_dll_path();
    
    py::class_<T2SEngine>(m, "T2SEngine")
        .def(py::init<>())
        .def("load_models", &T2SEngine::load_models,
             py::arg("encoder_path"),
             py::arg("fsd_path"),
             py::arg("sd_path"),
             py::arg("providers") = std::vector<std::string>{"CPUExecutionProvider"})
        .def("run_t2s", &T2SEngine::run_t2s,
             py::arg("ref_seq"),
             py::arg("ref_bert"),
             py::arg("text_seq"),
             py::arg("text_bert"),
             py::arg("ssl_content"))
        .def("set_max_steps", &T2SEngine::set_max_steps)
        .def_property_readonly("loaded", [](const T2SEngine& e) { return e.loaded_; });
    
    m.attr("__version__") = "0.1.0";
}
