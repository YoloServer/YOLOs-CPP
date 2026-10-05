/**
 * @file device_check.cpp
 * @brief Lists the inference devices this build can use and optionally benchmarks a model on them.
 *
 *   device_check                           list devices
 *   device_check model.onnx                benchmark the model on every listed device
 *   device_check model.onnx cpu auto       benchmark only the given device requests
 *   device_check model.onnx --runs 50      timed runs per device (default 20)
 *   device_check model.onnx --cache DIR    compiled-model cache directory (OpenVINO, CoreML)
 *
 * Paste the output into a bug report or into docs/guides/hardware.md when testing new hardware.
 */

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "yolos/tasks/detection.hpp"

int main(int argc, char** argv) {
    std::string modelPath;
    std::vector<std::string> requests;
    int runs = 20;
    std::string cacheDir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [model.onnx] [device ...] [--runs N] [--cache DIR]\n";
            return 0;
        }
        if (arg == "--cache" && i + 1 < argc) {
            cacheDir = argv[++i];
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::stoi(argv[++i]));
        } else if (modelPath.empty() && arg.size() > 5 && arg.substr(arg.size() - 5) == ".onnx") {
            modelPath = arg;
        } else {
            requests.push_back(arg);
        }
    }

    std::cout << "ONNX Runtime " << Ort::GetVersionString() << "\nExecution providers:";
    for (const auto& p : yolos::availableExecutionProviders()) std::cout << ' ' << p;
    std::cout << "\nDevices:";
    const auto devices = yolos::availableDevices();
    for (const auto& d : devices) std::cout << ' ' << d;
    std::cout << "\n";

    if (modelPath.empty()) return 0;
    if (requests.empty()) requests = devices;

    std::cout << "\nBenchmark: " << modelPath << " (" << runs << " runs, 640x640 input)\n";
    std::cout << std::left << std::setw(16) << "requested" << std::setw(18) << "used" << std::setw(12) << "load (ms)"
              << "mean (ms)\n";

    const cv::Mat image(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    int failures = 0;
    for (const auto& request : requests) {
        try {
            const auto t0 = std::chrono::steady_clock::now();
            yolos::DeviceConfig device(request);
            device.cacheDir = cacheDir;
            yolos::det::YOLODetector detector(modelPath, "", device);
            const auto t1 = std::chrono::steady_clock::now();

            detector.detect(image);  // warm-up: first run compiles kernels on some providers
            const auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < runs; ++i) detector.detect(image);
            const auto t3 = std::chrono::steady_clock::now();

            using ms = std::chrono::duration<double, std::milli>;
            std::cout << std::left << std::setw(16) << request << std::setw(18) << detector.getDevice()
                      << std::setw(12) << std::fixed << std::setprecision(1) << ms(t1 - t0).count()
                      << ms(t3 - t2).count() / runs << "\n";
        } catch (const std::exception& e) {
            ++failures;
            std::cout << std::left << std::setw(16) << request << "FAILED: " << e.what() << "\n";
        }
    }
    return failures == 0 ? 0 : 1;
}
