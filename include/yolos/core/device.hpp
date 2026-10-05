#pragma once

// ============================================================================
// Inference device selection
// ============================================================================
// Describes *where* a model should run (CPU, CUDA, OpenVINO, DirectML, ...) and
// resolves that request against the execution providers an ONNX Runtime build
// actually offers. This header has no ONNX Runtime dependency, so the logic can
// be unit-tested without a GPU or an ONNX Runtime installation.
//
// Backwards compatibility: DeviceConfig is implicitly constructible from bool, so
// every existing `YOLODetector(model, labels, /*useGPU=*/true)` call keeps compiling.
// ============================================================================

#include <algorithm>
#include <cctype>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace yolos {

/// @brief Where and how to run inference
struct DeviceConfig {
    /// Device request. One of:
    ///   "cpu"
    ///   "auto"                     best accelerator available, CPU if none works
    ///   "cuda" | "cuda:N"          NVIDIA CUDA
    ///   "tensorrt" | "tensorrt:N"  NVIDIA TensorRT (never picked by "auto": engine builds are slow)
    ///   "rocm" | "migraphx"        AMD (Linux)
    ///   "openvino[:TYPE]"          Intel; TYPE is GPU (default), NPU, CPU, or e.g. GPU.1
    ///   "dml" | "dml:N"            DirectML (Windows: any DirectX 12 GPU)
    ///   "coreml"                   Apple (GPU / Neural Engine)
    std::string device = "cpu";

    /// Intra-op threads for CPU work (0 = automatic)
    int numThreads = 0;

    /// Directory for compiled-model caches (OpenVINO, CoreML). Empty = provider default.
    std::string cacheDir;

    /// Raw provider options, passed through untouched for advanced tuning
    std::map<std::string, std::string> options;

    DeviceConfig() = default;
    /// true -> "auto", false -> "cpu" (keeps the old `bool useGPU` call sites working)
    DeviceConfig(bool useGpu) : device(useGpu ? "auto" : "cpu") {}
    /// Keeps `0` / `1` literals from being ambiguous between bool and const char*
    DeviceConfig(int useGpu) : DeviceConfig(useGpu != 0) {}
    DeviceConfig(const char* dev) : device(dev ? dev : "cpu") {}
    DeviceConfig(std::string dev) : device(std::move(dev)) {}
};

/// @brief A device request split into its parts, e.g. "openvino:GPU.1" -> {"openvino", "GPU.1", -1}
struct DeviceSpec {
    std::string provider;    ///< lower-case: cpu, auto, cuda, tensorrt, rocm, migraphx, openvino, dml, coreml
    std::string deviceType;  ///< text after ':' (original case), empty if none
    int index = -1;          ///< numeric text after ':' (cuda:1 -> 1), -1 if none
};

namespace detail {

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

}  // namespace detail

/// @brief Parse a device request. Unknown providers are kept as-is; resolveDevice() rejects them.
inline DeviceSpec parseDevice(const std::string& request) {
    DeviceSpec spec;
    const std::string text = detail::trim(request);
    const size_t colon = text.find(':');
    spec.provider = detail::toLower(detail::trim(text.substr(0, colon)));
    if (colon != std::string::npos) spec.deviceType = detail::trim(text.substr(colon + 1));

    // Friendly aliases
    if (spec.provider.empty()) spec.provider = "cpu";
    if (spec.provider == "gpu") spec.provider = "auto";
    if (spec.provider == "directml") spec.provider = "dml";
    if (spec.provider == "intel") spec.provider = "openvino";
    if (spec.provider == "apple") spec.provider = "coreml";

    if (!spec.deviceType.empty() &&
        std::all_of(spec.deviceType.begin(), spec.deviceType.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; })) {
        spec.index = std::stoi(spec.deviceType);
    }
    return spec;
}

/// @brief One concrete execution provider to try
struct DeviceCandidate {
    std::string provider;    ///< canonical key, see DeviceSpec::provider (never "auto")
    std::string ortName;     ///< ONNX Runtime provider name, e.g. "OpenVINOExecutionProvider"
    std::string deviceType;  ///< provider-specific device selector, e.g. "GPU" for OpenVINO
    int index = -1;
};

/// @brief ONNX Runtime provider name for a canonical provider key ("" if unknown)
inline std::string ortProviderName(const std::string& provider) {
    if (provider == "cpu") return "CPUExecutionProvider";
    if (provider == "cuda") return "CUDAExecutionProvider";
    if (provider == "tensorrt") return "TensorrtExecutionProvider";
    if (provider == "rocm") return "ROCMExecutionProvider";
    if (provider == "migraphx") return "MIGraphXExecutionProvider";
    if (provider == "openvino") return "OpenVINOExecutionProvider";
    if (provider == "dml") return "DmlExecutionProvider";
    if (provider == "coreml") return "CoreMLExecutionProvider";
    return "";
}

/// @brief Result of resolving a request against the available providers
struct DeviceResolution {
    /// Providers to try, best first. CPU is always the implicit final fallback and is not listed.
    std::vector<DeviceCandidate> candidates;
    /// Human-readable problems (unknown device, provider not in this ONNX Runtime build, ...)
    std::vector<std::string> warnings;
};

/// @brief Turn a device request into an ordered list of providers to try
/// @param availableProviders names from Ort::GetAvailableProviders()
inline DeviceResolution resolveDevice(const std::string& request, const std::vector<std::string>& availableProviders) {
    DeviceResolution out;
    const DeviceSpec spec = parseDevice(request);

    auto has = [&](const std::string& ortName) {
        return std::find(availableProviders.begin(), availableProviders.end(), ortName) != availableProviders.end();
    };
    auto makeCandidate = [](const std::string& provider, const std::string& deviceType, int index) {
        DeviceCandidate c;
        c.provider = provider;
        c.ortName = ortProviderName(provider);
        c.deviceType = deviceType;
        c.index = index;
        return c;
    };

    if (spec.provider == "cpu") return out;

    if (spec.provider == "auto") {
        // Order: most mature and fastest first. TensorRT is explicit-only.
        static const char* kOrder[] = {"cuda", "rocm", "migraphx", "openvino", "dml", "coreml"};
        for (const char* p : kOrder) {
            if (!has(ortProviderName(p))) continue;
            // OpenVINO also lists a CPU device; only ask for the GPU so "auto" means an accelerator
            out.candidates.push_back(makeCandidate(p, std::string(p) == "openvino" ? "GPU" : "", -1));
        }
        if (out.candidates.empty()) out.warnings.push_back("no GPU execution provider in this ONNX Runtime build");
        return out;
    }

    if (ortProviderName(spec.provider).empty()) {
        out.warnings.push_back("unknown device '" + request + "'");
        return out;
    }
    if (!has(ortProviderName(spec.provider))) {
        out.warnings.push_back(ortProviderName(spec.provider) + " is not part of this ONNX Runtime build");
        return out;
    }
    std::string type = spec.deviceType;
    if (spec.provider == "openvino" && type.empty()) type = "GPU";
    out.candidates.push_back(makeCandidate(spec.provider, spec.index >= 0 ? "" : type, spec.index));
    return out;
}

}  // namespace yolos
