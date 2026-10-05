#pragma once

// ============================================================================
// ONNX Runtime execution-provider setup
// ============================================================================
// Applies a DeviceConfig to Ort::SessionOptions. One implementation shared by every
// task class, so a new backend only ever needs to be added here.
// ============================================================================

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "yolos/core/device.hpp"

namespace yolos {

namespace detail {

/// Merge user-supplied raw options over the defaults we computed
inline std::unordered_map<std::string, std::string> withUserOptions(
    std::unordered_map<std::string, std::string> base, const DeviceConfig& cfg) {
    for (const auto& kv : cfg.options) base[kv.first] = kv.second;
    return base;
}

/// Append one provider to the session options. Throws Ort::Exception if the provider
/// cannot be created (driver missing, unsupported platform, ...).
inline std::string appendProvider(Ort::SessionOptions& so, const DeviceCandidate& c, const DeviceConfig& cfg) {
    const std::string idx = c.index >= 0 ? std::to_string(c.index) : "";
    const int deviceId = c.index >= 0 ? c.index : 0;

    if (c.provider == "cuda") {
        OrtCUDAProviderOptions o{};
        o.device_id = deviceId;
        so.AppendExecutionProvider_CUDA(o);
        return c.index >= 0 ? "cuda:" + idx : "cuda";
    }
    if (c.provider == "tensorrt") {
        OrtTensorRTProviderOptions o{};
        o.device_id = deviceId;
        so.AppendExecutionProvider_TensorRT(o);
        return c.index >= 0 ? "tensorrt:" + idx : "tensorrt";
    }
    if (c.provider == "rocm") {
        OrtROCMProviderOptions o{};
        o.device_id = deviceId;
        so.AppendExecutionProvider_ROCM(o);
        return c.index >= 0 ? "rocm:" + idx : "rocm";
    }
    if (c.provider == "migraphx") {
        OrtMIGraphXProviderOptions o{};
        o.device_id = deviceId;
        so.AppendExecutionProvider_MIGraphX(o);
        return c.index >= 0 ? "migraphx:" + idx : "migraphx";
    }
    if (c.provider == "openvino") {
        // "openvino:1" -> second GPU; otherwise the requested device type (GPU, NPU, CPU, GPU.1, ...)
        const std::string type = c.index >= 0 ? "GPU." + idx : (c.deviceType.empty() ? "GPU" : c.deviceType);
        std::unordered_map<std::string, std::string> opts{{"device_type", type}};
        if (!cfg.cacheDir.empty()) opts["cache_dir"] = cfg.cacheDir;
        so.AppendExecutionProvider_OpenVINO_V2(withUserOptions(std::move(opts), cfg));
        return "openvino:" + type;
    }
    if (c.provider == "dml") {
        // DirectML does not support memory-pattern optimisation or parallel execution
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.DisableMemPattern();
        so.AppendExecutionProvider("DML", withUserOptions({{"device_id", std::to_string(deviceId)}}, cfg));
        return c.index >= 0 ? "dml:" + idx : "dml";
    }
    if (c.provider == "coreml") {
        std::unordered_map<std::string, std::string> opts;
        if (!cfg.cacheDir.empty()) opts["ModelCacheDirectory"] = cfg.cacheDir;
        so.AppendExecutionProvider("CoreML", withUserOptions(std::move(opts), cfg));
        return "coreml";
    }
    throw Ort::Exception("unsupported provider '" + c.provider + "'", ORT_NOT_IMPLEMENTED);
}

}  // namespace detail

/// @brief Names of the execution providers compiled into the loaded ONNX Runtime
inline std::vector<std::string> availableExecutionProviders() { return Ort::GetAvailableProviders(); }

/// @brief Device requests that can be used on this machine's ONNX Runtime build, e.g. {"cpu", "openvino:GPU"}
/// A listed provider can still fail at load time if its driver is missing; sessions then fall back to CPU.
inline std::vector<std::string> availableDevices() {
    std::vector<std::string> out{"cpu"};
    const auto providers = availableExecutionProviders();
    for (const char* p : {"cuda", "tensorrt", "rocm", "migraphx", "openvino", "dml", "coreml"}) {
        if (std::find(providers.begin(), providers.end(), ortProviderName(p)) == providers.end()) continue;
        out.push_back(std::string(p) == "openvino" ? "openvino:GPU" : p);
    }
    return out;
}

/// @brief Configure session options for a device request
/// @param so options to configure (threads, graph optimisation and execution provider are set)
/// @param cfg device request
/// @param defaultThreads intra-op thread count used when cfg.numThreads is 0
/// @return the device actually selected ("cpu", "cuda", "openvino:GPU", ...). Never throws for a
///         missing accelerator: it logs why and returns "cpu".
inline std::string configureSession(Ort::SessionOptions& so, const DeviceConfig& cfg, int defaultThreads) {
    const int threads = cfg.numThreads > 0 ? cfg.numThreads : std::max(1, defaultThreads);
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const DeviceResolution resolved = resolveDevice(cfg.device, availableExecutionProviders());
    for (const auto& w : resolved.warnings) {
        std::cout << "[WARNING] " << w << ". Falling back to CPU." << std::endl;
    }

    for (const auto& candidate : resolved.candidates) {
        try {
            const std::string used = detail::appendProvider(so, candidate, cfg);
            std::cout << "[INFO] Inference device: " << used << std::endl;
            return used;
        } catch (const Ort::Exception& e) {
            std::cout << "[WARNING] " << candidate.ortName << " could not be enabled (" << e.what() << ")" << std::endl;
        }
    }

    std::cout << "[INFO] Inference device: CPU" << std::endl;
    return "cpu";
}

}  // namespace yolos
