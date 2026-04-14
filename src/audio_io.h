#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lunavox {

// PCM WAV read. On success `samples` holds mono float samples in [-1, 1]
// and `sample_rate` is set to the WAV header rate. Multi-channel input is
// downmixed to mono by averaging. Supports 16-bit PCM and 32-bit float.
bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate);

// 16-bit PCM WAV write. Creates parent directories if missing. Samples are
// clamped to [-1, 1] before quantization.
bool save_audio_file(const std::string & path, const std::vector<float> & samples, int sample_rate);

// Windowed-sinc resampler (Kaiser window). Higher quality than linear
// interpolation; used for reference-audio rate conversion.
bool resample_windowed_sinc(
    const float * input,
    int32_t input_len,
    int32_t input_rate,
    std::vector<float> & output,
    int32_t output_rate);

} // namespace lunavox
