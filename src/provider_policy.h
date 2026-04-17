#pragma once

#include <functional>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace lunavox {

enum class ort_session_role {
    cpu_only,
    decoder,
};

/**
 * One attempt at constructing an ORT session. The session factory walks
 * through the chain in order; if Ort::Session throws during construction
 * (partitioner bugs, driver issues, missing DLL, ...), the next candidate
 * is tried. The last entry is always a CPU fallback so the chain never
 * runs out of options on a machine that has ORT installed at all.
 *
 * Per-candidate knobs (opt_level, memory-pattern flags) exist because the
 * same role can need different session settings on different EPs: a CoreML
 * fallback often runs better with EXTENDED optimizations while the primary
 * CoreML path wants ALL; CPU decoders should still stay memory-stingy.
 */
struct ProviderCandidate {
    std::string name;  // Human-readable label, includes sub-config hints.
    GraphOptimizationLevel opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
    bool disable_mem_arena = false;
    bool disable_mem_pattern = false;
    bool disable_spinning = false;
    // Installs the EP into `opts`. Returning false (with err populated) causes
    // this candidate to be skipped without even attempting Ort::Session.
    std::function<bool(Ort::SessionOptions &, std::string &)> apply;
};

/**
 * Build the ordered fallback chain for the given role. The chain is the
 * single source of truth for (platform × role → EP preference); no other
 * site should need to know which EPs exist on which OS.
 */
std::vector<ProviderCandidate> build_provider_chain(ort_session_role role);

} // namespace lunavox
