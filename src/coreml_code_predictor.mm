#if !__has_feature(objc_arc)
#error This file must be compiled with ARC (-fobjc-arc)
#endif

#include "coreml_code_predictor.h"

#include <cstdio>
#include <cstring>

#if defined(__APPLE__)
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

namespace qwen3_tts {

struct CoreMLCodePredictor::Impl {
    MLModel * model = nil;
    NSURL * compiled_url = nil;
    MLMultiArray * seq_embd = nil;
    MLMultiArray * attn_mask = nil;
    MLMultiArray * selector = nil;

    int32_t n_steps = 0;
    int32_t max_seq = 16;
    int32_t hidden_size = 1024;
    int32_t vocab_size = 2048;

    std::string error;
};

static bool get_multiarray_shape_3d(MLFeatureDescription * desc,
                                    int32_t & d0,
                                    int32_t & d1,
                                    int32_t & d2,
                                    std::string & err) {
    if (!desc || desc.type != MLFeatureTypeMultiArray || !desc.multiArrayConstraint) {
        err = "Missing or invalid multi-array feature description";
        return false;
    }

    NSArray<NSNumber *> * shape = desc.multiArrayConstraint.shape;
    if (!shape || shape.count != 3) {
        err = "Expected rank-3 multi-array feature";
        return false;
    }

    d0 = (int32_t) shape[0].intValue;
    d1 = (int32_t) shape[1].intValue;
    d2 = (int32_t) shape[2].intValue;
    return true;
}

static MLMultiArray * create_multiarray(NSArray<NSNumber *> * shape,
                                        MLMultiArrayDataType dtype,
                                        std::string & err) {
    NSError * ns_error = nil;
    MLMultiArray * arr = [[MLMultiArray alloc] initWithShape:shape dataType:dtype error:&ns_error];
    if (!arr) {
        err = ns_error ? std::string([[ns_error localizedDescription] UTF8String])
                       : std::string("Failed to allocate MLMultiArray");
        return nil;
    }
    return arr;
}

CoreMLCodePredictor::CoreMLCodePredictor() : impl_(new Impl()) {}

CoreMLCodePredictor::~CoreMLCodePredictor() {
    unload();
    delete impl_;
    impl_ = nullptr;
}

bool CoreMLCodePredictor::load(const std::string & model_path, int32_t n_steps) {
    if (!impl_) {
        return false;
    }

    unload();

    @autoreleasepool {
        NSString * ns_model_path = [[NSString alloc] initWithUTF8String:model_path.c_str()];
        NSURL * model_url = [NSURL fileURLWithPath:ns_model_path];
        NSURL * load_url = model_url;

        NSString * ext = [[ns_model_path pathExtension] lowercaseString];
        if ([ext isEqualToString:@"mlpackage"] || [ext isEqualToString:@"mlmodel"]) {
            NSError * compile_error = nil;
            NSURL * compiled = [MLModel compileModelAtURL:model_url error:&compile_error];
            if (!compiled) {
                impl_->error = compile_error ? std::string([[compile_error localizedDescription] UTF8String])
                                             : std::string("Failed to compile CoreML model");
                return false;
            }
            impl_->compiled_url = compiled;
            load_url = compiled;
        }

        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        NSError * ns_error = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:load_url configuration:config error:&ns_error];
        if (!model) {
            impl_->error = ns_error ? std::string([[ns_error localizedDescription] UTF8String])
                                    : std::string("Failed to load CoreML model");
            return false;
        }

        int32_t seq_b = 0, seq_t = 0, seq_d = 0;
        int32_t out_b = 0, out_s = 0, out_v = 0;

        NSDictionary<NSString *, MLFeatureDescription *> * inputs = model.modelDescription.inputDescriptionsByName;
        NSDictionary<NSString *, MLFeatureDescription *> * outputs = model.modelDescription.outputDescriptionsByName;

        std::string shape_err;
        if (!get_multiarray_shape_3d(inputs[@"seq_embd"], seq_b, seq_t, seq_d, shape_err)) {
            impl_->error = "Invalid CoreML input seq_embd: " + shape_err;
            return false;
        }
        if (!get_multiarray_shape_3d(outputs[@"logits_all"], out_b, out_s, out_v, shape_err)) {
            impl_->error = "Invalid CoreML output logits_all: " + shape_err;
            return false;
        }

        if (seq_b != 1 || out_b != 1) {
            impl_->error = "CoreML model must use batch size 1";
            return false;
        }

        impl_->model = model;
        impl_->n_steps = n_steps;
        impl_->max_seq = seq_t;
        impl_->hidden_size = seq_d;
        impl_->vocab_size = out_v;

        if (out_s < n_steps) {
            impl_->error = "CoreML logits_all has fewer step heads than requested";
            unload();
            return false;
        }

        impl_->seq_embd = create_multiarray(@[ @1, @(impl_->max_seq), @(impl_->hidden_size) ],
                                            MLMultiArrayDataTypeFloat32,
                                            impl_->error);
        if (!impl_->seq_embd) {
            unload();
            return false;
        }

        impl_->attn_mask = create_multiarray(@[ @1, @1, @(impl_->max_seq), @(impl_->max_seq) ],
                                             MLMultiArrayDataTypeFloat32,
                                             impl_->error);
        if (!impl_->attn_mask) {
            unload();
            return false;
        }

        impl_->selector = create_multiarray(@[ @1, @(impl_->max_seq), @1 ],
                                            MLMultiArrayDataTypeFloat32,
                                            impl_->error);
        if (!impl_->selector) {
            unload();
            return false;
        }

        fprintf(stderr,
                "  CoreML code predictor loaded: max_seq=%d hidden=%d vocab=%d steps=%d (CPU+NE)\n",
                impl_->max_seq, impl_->hidden_size, impl_->vocab_size, impl_->n_steps);
    }

    return true;
}

void CoreMLCodePredictor::unload() {
    if (!impl_) {
        return;
    }

    @autoreleasepool {
        impl_->selector = nil;
        impl_->attn_mask = nil;
        impl_->seq_embd = nil;
        impl_->compiled_url = nil;
        impl_->model = nil;
    }

    impl_->error.clear();
    impl_->n_steps = 0;
    impl_->max_seq = 16;
    impl_->hidden_size = 1024;
    impl_->vocab_size = 2048;
}

bool CoreMLCodePredictor::is_loaded() const {
    return impl_ && impl_->model != nil;
}

const std::string & CoreMLCodePredictor::get_error() const {
    static const std::string empty;
    return impl_ ? impl_->error : empty;
}

bool CoreMLCodePredictor::predict_step(int32_t step_idx,
                                       const float * seq_embd,
                                       int32_t seq_len,
                                       int32_t hidden_size,
                                       std::vector<float> & logits_out) {
    if (!impl_ || impl_->model == nil) {
        if (impl_) impl_->error = "CoreML model not loaded";
        return false;
    }
    if (!seq_embd) {
        impl_->error = "seq_embd is null";
        return false;
    }
    if (step_idx < 0 || step_idx >= impl_->n_steps) {
        impl_->error = "step_idx out of range";
        return false;
    }
    if (hidden_size != impl_->hidden_size) {
        impl_->error = "hidden_size mismatch";
        return false;
    }
    if (seq_len <= 0 || seq_len > impl_->max_seq) {
        impl_->error = "seq_len out of range";
        return false;
    }

    @autoreleasepool {
        // Fill seq embeddings [1, T, D].
        float * seq_ptr = (float *) impl_->seq_embd.dataPointer;
        memset(seq_ptr, 0, (size_t)impl_->seq_embd.count * sizeof(float));

        NSArray<NSNumber *> * seq_strides = impl_->seq_embd.strides;
        const int64_t st = (int64_t) seq_strides[1].longLongValue;
        const int64_t sd = (int64_t) seq_strides[2].longLongValue;

        for (int32_t t = 0; t < seq_len; ++t) {
            const float * src = seq_embd + (size_t)t * hidden_size;
            for (int32_t d = 0; d < hidden_size; ++d) {
                seq_ptr[(int64_t)t * st + (int64_t)d * sd] = src[d];
            }
        }

        // Fill attention mask [1, 1, T, T] with additive values (0 or large negative).
        float * mask_ptr = (float *) impl_->attn_mask.dataPointer;
        NSArray<NSNumber *> * mask_strides = impl_->attn_mask.strides;
        const int64_t si = (int64_t) mask_strides[2].longLongValue;
        const int64_t sj = (int64_t) mask_strides[3].longLongValue;

        for (int32_t i = 0; i < impl_->max_seq; ++i) {
            for (int32_t j = 0; j < impl_->max_seq; ++j) {
                float v = -1e9f;
                if (i < seq_len && j < seq_len && j <= i) {
                    v = 0.0f;
                }
                mask_ptr[(int64_t)i * si + (int64_t)j * sj] = v;
            }
        }

        // Fill selector [1, T, 1] one-hot at last valid token row.
        float * sel_ptr = (float *) impl_->selector.dataPointer;
        memset(sel_ptr, 0, (size_t)impl_->selector.count * sizeof(float));
        NSArray<NSNumber *> * sel_strides = impl_->selector.strides;
        const int64_t sst = (int64_t) sel_strides[1].longLongValue;
        sel_ptr[(int64_t)(seq_len - 1) * sst] = 1.0f;

        NSError * ns_error = nil;
        NSDictionary<NSString *, MLFeatureValue *> * dict = @{
            @"seq_embd": [MLFeatureValue featureValueWithMultiArray:impl_->seq_embd],
            @"attn_mask": [MLFeatureValue featureValueWithMultiArray:impl_->attn_mask],
            @"selector": [MLFeatureValue featureValueWithMultiArray:impl_->selector],
        };
        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&ns_error];
        if (!provider) {
            impl_->error = ns_error ? std::string([[ns_error localizedDescription] UTF8String])
                                    : std::string("Failed to create CoreML input provider");
            return false;
        }

        id<MLFeatureProvider> pred = [impl_->model predictionFromFeatures:provider error:&ns_error];
        if (!pred) {
            impl_->error = ns_error ? std::string([[ns_error localizedDescription] UTF8String])
                                    : std::string("CoreML prediction failed");
            return false;
        }

        MLFeatureValue * fv = [pred featureValueForName:@"logits_all"];
        if (!fv || fv.type != MLFeatureTypeMultiArray || !fv.multiArrayValue) {
            impl_->error = "Missing logits_all output";
            return false;
        }

        MLMultiArray * logits = fv.multiArrayValue;
        if (logits.dataType != MLMultiArrayDataTypeFloat32) {
            impl_->error = "Expected float32 logits_all output";
            return false;
        }

        NSArray<NSNumber *> * shape = logits.shape;
        if (shape.count != 3 || shape[0].intValue != 1 || shape[1].intValue <= step_idx) {
            impl_->error = "Unexpected logits_all shape";
            return false;
        }

        NSArray<NSNumber *> * strides = logits.strides;
        const int64_t ss = (int64_t) strides[1].longLongValue;
        const int64_t sv = (int64_t) strides[2].longLongValue;

        logits_out.resize(impl_->vocab_size);
        const float * logits_ptr = (const float *) logits.dataPointer;
        for (int32_t v = 0; v < impl_->vocab_size; ++v) {
            logits_out[v] = logits_ptr[(int64_t)step_idx * ss + (int64_t)v * sv];
        }
    }

    return true;
}

} // namespace qwen3_tts

#else

namespace qwen3_tts {

CoreMLCodePredictor::CoreMLCodePredictor() {}
CoreMLCodePredictor::~CoreMLCodePredictor() {}

bool CoreMLCodePredictor::load(const std::string &, int32_t) { return false; }
void CoreMLCodePredictor::unload() {}
bool CoreMLCodePredictor::is_loaded() const { return false; }
const std::string & CoreMLCodePredictor::get_error() const {
    static const std::string err = "CoreML predictor only supported on Apple platforms";
    return err;
}
bool CoreMLCodePredictor::predict_step(int32_t, const float *, int32_t, int32_t, std::vector<float> &) {
    return false;
}

} // namespace qwen3_tts

#endif
