#include "gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <string>

int main() {
    // Load GGUF
    qwen3_tts::GGUFLoader loader;
    if (!loader.open("models/qwen3-tts-tokenizer-f16.gguf")) {
        fprintf(stderr, "Failed to open GGUF\n");
        return 1;
    }
    
    // Find codebook and usage tensors
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    struct ggml_tensor * codebook_meta = ggml_get_tensor(meta_ctx, "tok_dec.vq_first.0.codebook");
    struct ggml_tensor * usage_meta = ggml_get_tensor(meta_ctx, "tok_dec.vq_first.0.usage");
    
    if (!codebook_meta || !usage_meta) {
        fprintf(stderr, "Failed to find tensors\n");
        return 1;
    }
    
    printf("Codebook shape: [%lld, %lld]\n", (long long)codebook_meta->ne[0], (long long)codebook_meta->ne[1]);
    printf("Usage shape: [%lld]\n", (long long)usage_meta->ne[0]);
    
    // Create context for tensors
    size_t ctx_size = ggml_tensor_overhead() * 10;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    
    struct ggml_tensor * codebook = ggml_dup_tensor(ctx, codebook_meta);
    ggml_set_name(codebook, "codebook");
    struct ggml_tensor * usage = ggml_dup_tensor(ctx, usage_meta);
    ggml_set_name(usage, "usage");
    
    // Load data
    std::map<std::string, struct ggml_tensor *> tensors;
    tensors["tok_dec.vq_first.0.codebook"] = codebook;
    tensors["tok_dec.vq_first.0.usage"] = usage;
    
    ggml_backend_buffer_t buffer = nullptr;
    std::string error;
    if (!qwen3_tts::load_tensor_data_from_file("models/qwen3-tts-tokenizer-f16.gguf", 
                                                loader.get_ctx(), ctx, tensors, buffer, error)) {
        fprintf(stderr, "Failed to load tensor data: %s\n", error.c_str());
        return 1;
    }
    
    // Normalize codebook
    int64_t codebook_dim = codebook->ne[0];
    int64_t codebook_size = codebook->ne[1];
    ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data;
    float * usage_data = (float *)usage->data;
    
    printf("\nBefore normalization:\n");
    printf("Entry 1221 first 5 values: ");
    for (int i = 0; i < 5; ++i) {
        int64_t idx = i + 1221 * codebook_dim;
        printf("%.4f ", ggml_fp16_to_fp32(cb_data[idx]));
    }
    printf("\n");
    printf("Usage[1221] = %.6f\n", usage_data[1221]);
    
    // Normalize
    for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
        float u = usage_data[emb_idx];
        if (u < 1e-5f) u = 1e-5f;
        float inv_u = 1.0f / u;
        
        for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
            int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
            float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
            cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
        }
    }
    
    printf("\nAfter normalization:\n");
    printf("Entry 1221 first 10 values: ");
    for (int i = 0; i < 10; ++i) {
        int64_t idx = i + 1221 * codebook_dim;
        printf("%.4f ", ggml_fp16_to_fp32(cb_data[idx]));
    }
    printf("\n");
    
    // Expected: [ 17.672094  -19.23553    -6.0868597   1.6018629  10.926484   18.936895
    //            -29.986347  -22.713736   23.90827    -8.511064 ]
    
    // Now test ggml_get_rows
    printf("\nTesting ggml_get_rows...\n");
    
    // Create compute context
    size_t compute_size = 1024 * 1024;
    std::vector<uint8_t> compute_buf(compute_size);
    struct ggml_init_params compute_params = {
        /*.mem_size   =*/ compute_size,
        /*.mem_buffer =*/ compute_buf.data(),
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_compute = ggml_init(compute_params);
    
    // Create index tensor
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, 1);
    ((int32_t *)indices->data)[0] = 1221;
    
    // Get rows
    struct ggml_tensor * row = ggml_get_rows(ctx_compute, codebook, indices);
    
    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx_compute);
    ggml_build_forward_expand(gf, row);
    
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_graph_compute(backend, gf);
    
    printf("Row shape: [%lld, %lld]\n", (long long)row->ne[0], (long long)row->ne[1]);
    printf("Row first 10 values: ");
    float * row_data = (float *)row->data;
    for (int i = 0; i < 10; ++i) {
        printf("%.4f ", row_data[i]);
    }
    printf("\n");
    
    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(ctx_compute);
    
    return 0;
}
