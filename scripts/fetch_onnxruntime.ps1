<#
.SYNOPSIS
    Fetch an ONNX Runtime build for one hardware backend (Windows).

.DESCRIPTION
    Produces a folder with include\ and lib\ that can be passed to CMake:

        .\scripts\fetch_onnxruntime.ps1 -Backend openvino -Version 1.20.0 -OutDir third_party
        cmake -S . -B build -DONNXRUNTIME_DIR=third_party\onnxruntime-openvino-1.20.0

    Backends:
        cpu       Microsoft release
        cuda      Microsoft release, NVIDIA GPU (needs CUDA 12 + cuDNN 9)
        directml  Microsoft DirectML NuGet package: any DirectX 12 GPU (Intel, AMD, NVIDIA)
        openvino  Intel OpenVINO build: Intel iGPU, Arc, NPU. Assembled from the
                  onnxruntime-openvino and openvino PyPI wheels (no Python needed).

    Headers and the import library always come from the matching Microsoft release.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "cuda", "directml", "openvino")]
    [string]$Backend,
    [string]$Version = "1.20.0",
    [string]$OutDir = "."
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is much faster without the progress bar

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$Dest = Join-Path $OutDir "onnxruntime-$Backend-$Version"
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ort-fetch-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null

if (Test-Path (Join-Path $Dest "include\onnxruntime_cxx_api.h")) {
    Write-Host "Already present: $Dest"
    return
}

# Download a file and unzip it (wheels and nupkgs are zip files) into $Into
function Get-Zip([string]$Url, [string]$Into) {
    $zip = Join-Path $Tmp ([System.IO.Path]::GetRandomFileName() + ".zip")
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $Into -Force
    Remove-Item $zip
}

# URL of the Windows x64 wheel of a PyPI package (any CPython tag; the DLLs are the same)
function Get-WheelUrl([string]$Package, [string]$Ver) {
    $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$Package/$Ver/json"
    $wheel = $meta.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
    if (-not $wheel) { throw "No win_amd64 wheel for $Package $Ver" }
    return $wheel.url
}

# Microsoft CPU release: headers + onnxruntime.lib
function Get-MicrosoftRelease([string]$Pkg, [string]$Into) {
    Get-Zip "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Pkg.zip" $Into
    return (Join-Path $Into $Pkg)
}

try {
    switch ($Backend) {
        "cpu" {
            $src = Get-MicrosoftRelease "onnxruntime-win-x64-$Version" $Tmp
            Move-Item $src $Dest
        }
        "cuda" {
            $src = Get-MicrosoftRelease "onnxruntime-win-x64-gpu-$Version" $Tmp
            Move-Item $src $Dest
        }
        "directml" {
            $pkg = Join-Path $Tmp "ortdml"
            Get-Zip "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$Version" $pkg
            New-Item -ItemType Directory -Force -Path "$Dest\lib" | Out-Null
            Copy-Item "$pkg\build\native\include" "$Dest\include" -Recurse
            Copy-Item "$pkg\runtimes\win-x64\native\*" "$Dest\lib" -Include *.dll, *.lib

            # Ship DirectML itself next to the runtime: the copy in System32 can be older than ORT expects
            $dml = Join-Path $Tmp "dml"
            Get-Zip "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4" $dml
            Copy-Item "$dml\bin\x64-win\DirectML.dll" "$Dest\lib"
        }
        "openvino" {
            # ONNX Runtime + OpenVINO provider (onnxruntime-openvino) and the OpenVINO runtime with its
            # Intel GPU / NPU / CPU plugins (openvino). 1.20.0 was built against OpenVINO 2024.4.0.
            $ovVersion = "2024.4.0"
            $ort = Join-Path $Tmp "ort"
            $ov = Join-Path $Tmp "ov"
            Get-Zip (Get-WheelUrl "onnxruntime-openvino" $Version) $ort
            Get-Zip (Get-WheelUrl "openvino" $ovVersion) $ov

            $ms = Get-MicrosoftRelease "onnxruntime-win-x64-$Version" (Join-Path $Tmp "ms")
            New-Item -ItemType Directory -Force -Path "$Dest\lib" | Out-Null
            Copy-Item "$ms\include" "$Dest\include" -Recurse
            Copy-Item "$ms\lib\onnxruntime.lib" "$Dest\lib"                       # import library
            Copy-Item "$ort\onnxruntime\capi\*.dll" "$Dest\lib"                   # runtime + OpenVINO provider
            Copy-Item "$ov\openvino\libs\*.dll" "$Dest\lib"                       # OpenVINO runtime + plugins
        }
    }
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}

Write-Host "ONNX Runtime ($Backend, $Version) ready: $Dest"
Write-Host "Use it with:  cmake -S . -B build -DONNXRUNTIME_DIR=$Dest"
