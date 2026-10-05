/**
 * @file test_device_selection.cpp
 * @brief Tests for device selection (CPU / CUDA / OpenVINO / DirectML / CoreML / ...)
 *
 * The resolution logic is pure and tested against fake provider lists, so these tests need
 * no GPU. The session tests run against the synthetic ONNX models and check that every
 * device request ends in a working session: an accelerator that is missing from this
 * ONNX Runtime build must fall back to the CPU instead of failing.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "yolos/yolos.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using yolos::DeviceConfig;
using yolos::parseDevice;
using yolos::resolveDevice;

namespace {

const std::vector<std::string> kCpuOnly = {"CPUExecutionProvider"};

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

}  // namespace

// ============================================================================
// Parsing
// ============================================================================

TEST(DeviceParse, SplitsProviderAndType) {
    auto s = parseDevice("OpenVINO:GPU.1");
    EXPECT_EQ(s.provider, "openvino");
    EXPECT_EQ(s.deviceType, "GPU.1");
    EXPECT_EQ(s.index, -1);
}

TEST(DeviceParse, NumericSuffixIsAnIndex) {
    auto s = parseDevice("cuda:2");
    EXPECT_EQ(s.provider, "cuda");
    EXPECT_EQ(s.index, 2);
}

TEST(DeviceParse, AliasesAndWhitespace) {
    EXPECT_EQ(parseDevice("  GPU ").provider, "auto");
    EXPECT_EQ(parseDevice("DirectML").provider, "dml");
    EXPECT_EQ(parseDevice("intel").provider, "openvino");
    EXPECT_EQ(parseDevice("apple").provider, "coreml");
    EXPECT_EQ(parseDevice("").provider, "cpu");
}

TEST(DeviceConfigConversions, KeepsTheOldBoolApi) {
    EXPECT_EQ(DeviceConfig(true).device, "auto");
    EXPECT_EQ(DeviceConfig(false).device, "cpu");
    EXPECT_EQ(DeviceConfig(1).device, "auto");
    EXPECT_EQ(DeviceConfig(0).device, "cpu");  // 0 must not be read as a null const char*
    EXPECT_EQ(DeviceConfig("cuda:1").device, "cuda:1");
    EXPECT_EQ(DeviceConfig(std::string("dml")).device, "dml");
    EXPECT_EQ(DeviceConfig().device, "cpu");

    const DeviceConfig fromBool = true;  // what `YOLODetector(model, labels, true)` relies on
    EXPECT_EQ(fromBool.device, "auto");
}

// ============================================================================
// Resolution
// ============================================================================

TEST(DeviceResolve, CpuNeedsNoProvider) {
    auto r = resolveDevice("cpu", {"CPUExecutionProvider", "CUDAExecutionProvider"});
    EXPECT_TRUE(r.candidates.empty());
    EXPECT_TRUE(r.warnings.empty());
}

TEST(DeviceResolve, AutoWithoutAcceleratorWarnsAndUsesCpu) {
    auto r = resolveDevice("auto", kCpuOnly);
    EXPECT_TRUE(r.candidates.empty());
    EXPECT_EQ(r.warnings.size(), 1u);
}

TEST(DeviceResolve, AutoPrefersCudaOverOpenVino) {
    auto r = resolveDevice("auto", {"OpenVINOExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"});
    ASSERT_EQ(r.candidates.size(), 2u);
    EXPECT_EQ(r.candidates[0].provider, "cuda");
    EXPECT_EQ(r.candidates[1].provider, "openvino");
    EXPECT_EQ(r.candidates[1].deviceType, "GPU");  // "auto" asks OpenVINO for the GPU, not its CPU device
}

TEST(DeviceResolve, AutoPicksTheProviderOfEachPlatform) {
    EXPECT_EQ(resolveDevice("auto", {"DmlExecutionProvider", "CPUExecutionProvider"}).candidates.at(0).provider, "dml");
    EXPECT_EQ(resolveDevice("auto", {"CoreMLExecutionProvider", "CPUExecutionProvider"}).candidates.at(0).provider,
              "coreml");
    EXPECT_EQ(resolveDevice("auto", {"MIGraphXExecutionProvider", "ROCMExecutionProvider"}).candidates.at(0).provider,
              "rocm");
}

TEST(DeviceResolve, AutoNeverPicksTensorRt) {
    auto r = resolveDevice("auto", {"TensorrtExecutionProvider", "CPUExecutionProvider"});
    EXPECT_TRUE(r.candidates.empty());
}

TEST(DeviceResolve, ExplicitProviderIsHonoured) {
    auto r = resolveDevice("tensorrt", {"TensorrtExecutionProvider", "CUDAExecutionProvider"});
    ASSERT_EQ(r.candidates.size(), 1u);
    EXPECT_EQ(r.candidates[0].provider, "tensorrt");

    r = resolveDevice("cuda:1", {"CUDAExecutionProvider"});
    ASSERT_EQ(r.candidates.size(), 1u);
    EXPECT_EQ(r.candidates[0].index, 1);
}

TEST(DeviceResolve, OpenVinoDeviceTypes) {
    const std::vector<std::string> ov = {"OpenVINOExecutionProvider"};
    EXPECT_EQ(resolveDevice("openvino", ov).candidates.at(0).deviceType, "GPU");
    EXPECT_EQ(resolveDevice("openvino:NPU", ov).candidates.at(0).deviceType, "NPU");
    EXPECT_EQ(resolveDevice("openvino:GPU.1", ov).candidates.at(0).deviceType, "GPU.1");
    EXPECT_EQ(resolveDevice("openvino:1", ov).candidates.at(0).index, 1);
}

TEST(DeviceResolve, UnavailableProviderWarnsInsteadOfFailing) {
    auto r = resolveDevice("openvino:GPU", kCpuOnly);
    EXPECT_TRUE(r.candidates.empty());
    ASSERT_EQ(r.warnings.size(), 1u);
    EXPECT_NE(r.warnings[0].find("OpenVINOExecutionProvider"), std::string::npos);
}

TEST(DeviceResolve, UnknownDeviceWarns) {
    auto r = resolveDevice("quantum", {"CUDAExecutionProvider"});
    EXPECT_TRUE(r.candidates.empty());
    EXPECT_EQ(r.warnings.size(), 1u);
}

// ============================================================================
// Real sessions
// ============================================================================

class DeviceSessionTest : public ::testing::Test {
protected:
    static std::string model() { return std::string(XSTRING(BASE_PATH_API)) + "models/det_dynamic.onnx"; }

    static void SetUpTestSuite() {
        ASSERT_TRUE(fs::exists(model())) << "Run: python3 tests/api/make_synthetic_models.py tests/api/models";
    }
};

TEST_F(DeviceSessionTest, CpuIsTheDefault) {
    yolos::det::YOLODetector detector(model(), "");
    EXPECT_EQ(detector.getDevice(), "cpu");
}

TEST_F(DeviceSessionTest, MissingAcceleratorFallsBackToCpu) {
    const auto providers = yolos::availableExecutionProviders();
    // Request a provider this ONNX Runtime build does not contain: must still give a working CPU session
    for (const char* request : {"openvino:GPU", "dml", "coreml", "rocm", "quantum"}) {
        const std::string ort = yolos::ortProviderName(yolos::parseDevice(request).provider);
        if (!ort.empty() && contains(providers, ort)) continue;  // present here: exercised by hardware tests
        yolos::det::YOLODetector detector(model(), "", request);
        EXPECT_EQ(detector.getDevice(), "cpu") << request;
    }
}

TEST_F(DeviceSessionTest, BoolTrueStillWorks) {
    yolos::det::YOLODetector detector(model(), "", true);  // pre-existing call style
    EXPECT_FALSE(detector.getDevice().empty());
}

TEST_F(DeviceSessionTest, AvailableDevicesAlwaysIncludesCpu) {
    const auto devices = yolos::availableDevices();
    ASSERT_FALSE(devices.empty());
    EXPECT_EQ(devices.front(), "cpu");
}
