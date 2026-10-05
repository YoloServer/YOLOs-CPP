# Hardware Acceleration

YOLOs-CPP runs on the CPU by default and can use a GPU or NPU through ONNX Runtime
*execution providers*. You choose the hardware with a device name; if it is not
available the library falls back to the CPU instead of failing.

## Choosing a device

Every task class (`YOLODetector`, `YOLOSegDetector`, `YOLOPoseDetector`, `YOLOOBBDetector`,
`YOLOClassifier`, `YOLODepthEstimator`, `YOLOE*`) takes a `DeviceConfig` where it used to take
`bool useGPU`:

```cpp
#include "yolos/yolos.hpp"

yolos::det::YOLODetector cpu    ("model.onnx", "coco.names", "cpu");
yolos::det::YOLODetector best   ("model.onnx", "coco.names", "auto");          // best accelerator, else CPU
yolos::det::YOLODetector nvidia ("model.onnx", "coco.names", "cuda:1");        // second NVIDIA GPU
yolos::det::YOLODetector intel  ("model.onnx", "coco.names", "openvino:GPU");  // Intel iGPU / Arc
yolos::det::YOLODetector npu    ("model.onnx", "coco.names", "openvino:NPU");  // Intel NPU
yolos::det::YOLODetector win    ("model.onnx", "coco.names", "dml");           // any DirectX 12 GPU
yolos::det::YOLODetector apple  ("model.onnx", "coco.names", "coreml");        // Apple GPU / Neural Engine

std::cout << best.getDevice();   // what is actually used: "cpu", "cuda", "openvino:GPU", ...
```

| Device string | Hardware | ONNX Runtime provider |
|---|---|---|
| `cpu` | Any CPU | `CPUExecutionProvider` |
| `auto` | Best accelerator this build offers, CPU if none works | see order below |
| `cuda`, `cuda:N` | NVIDIA GPU | `CUDAExecutionProvider` |
| `tensorrt`, `tensorrt:N` | NVIDIA GPU (TensorRT) | `TensorrtExecutionProvider` |
| `rocm`, `migraphx` | AMD GPU (Linux) | `ROCMExecutionProvider`, `MIGraphXExecutionProvider` |
| `openvino`, `openvino:TYPE` | Intel iGPU, Arc, NPU, CPU. `TYPE` is `GPU` (default), `NPU`, `CPU` or `GPU.1` | `OpenVINOExecutionProvider` |
| `dml`, `dml:N` | Any DirectX 12 GPU on Windows (Intel, AMD, NVIDIA) | `DmlExecutionProvider` |
| `coreml` | Apple Silicon GPU and Neural Engine | `CoreMLExecutionProvider` |

`auto` tries, in order: CUDA, ROCm, MIGraphX, OpenVINO (GPU), DirectML, CoreML. TensorRT is never
chosen automatically because its first-run engine build takes minutes; request it explicitly.

`DeviceConfig` also carries `numThreads`, `cacheDir` (compiled-model cache for OpenVINO and CoreML)
and a free-form `options` map that is passed to the provider unchanged:

```cpp
yolos::DeviceConfig cfg("openvino:GPU");
cfg.cacheDir = "/var/cache/yolos";          // skip the multi-second GPU compile on the next start
cfg.options["precision"] = "FP16";          // raw OpenVINO provider option
yolos::det::YOLODetector detector("model.onnx", "coco.names", cfg);
```

### Compatibility with `bool useGPU`

Existing code keeps compiling: `bool`, `int`, `const char*` and `std::string` convert to
`DeviceConfig`. `true` now means `"auto"` (it used to mean CUDA only) and `false` means `"cpu"`.
`getDevice()` now returns the provider name (`"cuda"`, `"openvino:GPU"`) where it used to return
`"gpu"`.

### What happens when the hardware is missing

The library never throws because an accelerator is absent:

1. The requested provider is not part of the loaded ONNX Runtime build: a warning is printed and the
   CPU is used.
2. The provider is present but cannot start (missing driver, device not exposed to the container):
   the same.
3. The provider starts but fails while loading the model: the session is created again on the CPU.

Each case logs why, and `getDevice()` returns `"cpu"`. Check it if you need to know.

### Listing what is available

```cpp
for (const auto& d : yolos::availableDevices()) std::cout << d << "\n";   // cpu, openvino:GPU, ...
```

The `device_check` program does this and can benchmark a model on every device:

```bash
./build/device_check                       # list devices
./build/device_check model.onnx            # time the model on each device
./build/device_check model.onnx cpu auto   # only these requests
```

A device that is listed can still fail to start (see above); `device_check model.onnx` shows
what each request really resolved to.

## Getting an ONNX Runtime with your hardware's provider

Providers are compiled into ONNX Runtime, so you need the build that contains yours. The library
code is the same for all of them; only the ONNX Runtime you link against changes.

| Hardware | Where the ONNX Runtime comes from | Fetch with |
|---|---|---|
| CPU | Microsoft release | `scripts/fetch_onnxruntime.sh cpu` |
| NVIDIA | Microsoft GPU release (needs CUDA 12 + cuDNN 9) | `scripts/fetch_onnxruntime.sh cuda` |
| Intel (Linux) | `onnxruntime-openvino` wheel: bundles OpenVINO and the Intel GPU/NPU plugins | `scripts/fetch_onnxruntime.sh openvino` |
| Intel (Windows) | `onnxruntime-openvino` + `openvino` wheels | `scripts/fetch_onnxruntime.ps1 openvino` |
| Any GPU on Windows | Microsoft DirectML NuGet package | `scripts/fetch_onnxruntime.ps1 directml` |
| Apple Silicon | Microsoft macOS release (includes CoreML) | `scripts/fetch_onnxruntime.sh coreml` |
| AMD (Linux) | Build ONNX Runtime from source with `--use_rocm` or `--use_migraphx` | not scripted yet |

```bash
scripts/fetch_onnxruntime.sh openvino 1.20.0 third_party
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_DIR=third_party/onnxruntime-openvino-1.20.0
cmake --build build --target device_check
./build/device_check
```

## Running in Docker

Containers do not see the GPU unless you pass it in. `Dockerfile.openvino` builds and tests
YOLOs-CPP with the OpenVINO provider and the Intel GPU drivers:

```bash
docker build -t yolos-cpp:openvino -f Dockerfile.openvino .
```

**Linux host** (Intel iGPU or Arc exposed as `/dev/dri`):

```bash
docker run --rm --device /dev/dri \
    --group-add "$(stat -c %g /dev/dri/render* | head -1)" \
    yolos-cpp:openvino
```

**Windows host, Docker Desktop with the WSL2 backend.** The GPU is exposed through `/dev/dxg`; its
user-mode libraries live in Docker Desktop's `/usr/lib/wsl` and must be mounted:

```bash
docker run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro yolos-cpp:openvino
```

In Git Bash prefix the command with `MSYS_NO_PATHCONV=1`, otherwise `/usr/lib/wsl` is rewritten to a
Windows path. PowerShell needs no prefix.

The default command runs `device_check` on a synthetic model, then the device tests. In the
output, the `used` column of the `openvino:GPU` row should say `openvino:GPU`. If it says `cpu`, the
container could not reach the GPU; look for the warning above it.

| Hardware | Docker support |
|---|---|
| NVIDIA | Yes: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (`docker run --gpus all`), see `Dockerfile` |
| Intel | Yes: `/dev/dri` on Linux, `/dev/dxg` with WSL2 on Windows |
| AMD | Linux hosts only (`--device /dev/kfd --device /dev/dri`) with a ROCm image; not available on Windows |
| Apple Silicon | **No.** Docker on macOS runs a Linux VM with no GPU or Neural Engine access, so containers are CPU-only. Run natively. |

## Native installation

Native builds avoid Docker's GPU limits and are the only option for Apple Silicon.

### Windows (PowerShell)

Needs the Visual Studio 2022 Build Tools (C++ workload), CMake and OpenCV. This is what the Intel and
DirectML results below were built with:

```powershell
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
# OpenCV: the prebuilt "opencv-4.x-windows.exe" from https://github.com/opencv/opencv/releases (self-extracting)

# In a "x64 Native Tools Command Prompt for VS 2022", or after running vcvars64.bat:
.\scripts\fetch_onnxruntime.ps1 -Backend openvino -Version 1.20.0 -OutDir third_party   # or: -Backend directml -Version 1.20.1
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=C:\opencv\build -DONNXRUNTIME_DIR=third_party\onnxruntime-openvino-1.20.0
cmake --build build --config Release --target device_check
$env:PATH = "C:\opencv\build\x64\vc16\bin;$env:PATH"      # OpenCV DLLs
.\build\Release\device_check.exe model.onnx
```

The CMake build copies every DLL of the ONNX Runtime folder (`onnxruntime.dll` plus the OpenVINO or
DirectML provider DLLs) next to the executables, so nothing else needs installing besides the GPU
driver. See also the [Windows 11 Guide](../YOLOs-CPP_on_Windows_11.md).

### Linux / macOS

```bash
scripts/fetch_onnxruntime.sh <cpu|cuda|openvino|coreml> 1.20.0 third_party
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=third_party/onnxruntime-<backend>-1.20.0
cmake --build build
```

On Linux the Intel GPU also needs the Intel compute runtime (`intel-opencl-icd`, `libze-intel-gpu1`);
`Dockerfile.openvino` shows the package list.

## Support matrix

| Hardware | Device | Platform | Status |
|---|---|---|---|
| CPU | `cpu` | Linux (Docker), x86-64 | Tested |
| Intel Iris Xe (11th gen, `0x9a49`) | `openvino:GPU` | Docker Desktop on Windows (WSL2) | Tested: `yolo11n` 640x640 runs in 16 ms/frame on the iGPU vs 114 ms on the CPU in the same container (see below) |
| Intel Iris Xe | `openvino:GPU` | Windows, native | Tested: `yolo11n` 18 ms/frame on the iGPU vs 83 ms on the CPU |
| Intel Arc, Core Ultra iGPU / NPU | `openvino` | Linux, Windows | Untested |
| NVIDIA | `cuda` | Linux, Windows | Code path from earlier releases; not re-tested with `DeviceConfig` |
| NVIDIA | `tensorrt` | Linux, Windows | Untested |
| AMD | `rocm`, `migraphx` | Linux | Untested |
| Intel Iris Xe | `dml` | Windows, native | Tested: `yolo11n` 44 ms/frame on the iGPU vs 94 ms on the CPU (OpenVINO is faster on Intel) |
| AMD, NVIDIA | `dml` | Windows | Untested |
| Apple Silicon | `coreml` | macOS | Untested |

"Untested" means the code compiles and the fallback to CPU is tested, but nobody has run it on that
hardware. Please report results.

## Testing on your hardware

The device tests need no GPU: they check the selection logic against fake provider lists and that
every request ends in a working session (falling back to the CPU where needed).

```bash
tests/test_api.sh                       # all API tests, including test_device_selection
cd tests/build && ./test_device_selection
```

To test real hardware, build against the ONNX Runtime for your provider (see above) and run:

```bash
./build/device_check tests/api/models/det_dynamic.onnx
```

The synthetic model proves the provider starts and runs, but its timings mean nothing: it is too
small for a GPU to pay off. For performance, pass a real model such as `yolo11n.onnx`.

When reporting a result, include the full output of `device_check`, your OS, GPU model and driver
version, and how you ran it (Docker or native). Add a row to the matrix above in your pull request.

Reference result for the Iris Xe row (Docker Desktop, WSL2, `yolo11n.onnx`, 30 runs, 640x640 input):

```
requested       used              load (ms)   mean (ms)
cpu             cpu               365.7       113.7
openvino:GPU    openvino:GPU      11101.0     16.6
```

## Troubleshooting

**`Specified device - GPU is not available` (OpenVINO).** The container or process cannot reach the
GPU. In Docker, check the `--device` flag and, on WSL2, the `/usr/lib/wsl` mount. `clinfo -l` inside
the container should list the Intel GPU. On native Windows, update the Intel graphics driver.

**`<provider> is not part of this ONNX Runtime build`.** You linked an ONNX Runtime without that
provider. Fetch the right one with `scripts/fetch_onnxruntime.sh`.

**The first run on a GPU takes seconds.** Providers compile the model for the device on load. Set
`DeviceConfig::cacheDir` (or `device_check --cache DIR`) to reuse the compiled model. Measured on an
Iris Xe with `yolo11n`: first load 10.8 s, next loads 0.7 s.

**Dynamic input shapes are slow on Intel GPUs.** Export the model with a fixed input size
(`imgsz=640`, no `dynamic`).
