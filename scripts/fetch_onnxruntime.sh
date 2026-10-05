#!/usr/bin/env bash
# ============================================================================
# Fetch an ONNX Runtime build for one hardware backend (Linux / macOS)
# ============================================================================
# Produces a directory with include/ and lib/ that can be passed to CMake:
#
#   scripts/fetch_onnxruntime.sh openvino 1.20.0 third_party
#   cmake -S . -B build -DONNXRUNTIME_DIR=third_party/onnxruntime-openvino-1.20.0
#
# Backends:
#   cpu       Microsoft release                     (Linux, macOS)
#   cuda      Microsoft release, NVIDIA GPU         (Linux x64)
#   openvino  Intel OpenVINO build: Intel iGPU, Arc, NPU, CPU (Linux x64)
#   coreml    Microsoft release; includes CoreML   (macOS: Apple GPU / Neural Engine)
#
# The OpenVINO build comes from the `onnxruntime-openvino` PyPI wheel, which bundles the
# OpenVINO runtime and the Intel GPU/NPU plugins. Only `pip` is needed, no Python code runs.
# Headers always come from the matching Microsoft release.
# ============================================================================
set -euo pipefail

BACKEND="${1:-}"
VERSION="${2:-1.20.0}"
OUT_ROOT="${3:-.}"

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}
[[ -z "$BACKEND" || "$BACKEND" == "-h" || "$BACKEND" == "--help" ]] && usage 0

case "$(uname -s)" in
Linux*)  OS=linux ;;
Darwin*) OS=osx ;;
*) echo "Unsupported OS $(uname -s). On Windows use scripts/fetch_onnxruntime.ps1" >&2; exit 1 ;;
esac
case "$(uname -m)" in
x86_64|amd64)  ARCH=x64 ;;
aarch64|arm64) ARCH=aarch64 ;;
*) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

MS_BASE="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}"
DEST="$OUT_ROOT/onnxruntime-${BACKEND}-${VERSION}"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
DEST="$OUT_ROOT/onnxruntime-${BACKEND}-${VERSION}"

if [[ -f "$DEST/include/onnxruntime_cxx_api.h" ]]; then
    echo "Already present: $DEST"
    echo "$DEST"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Download + unpack a Microsoft release archive into $1
fetch_microsoft() {
    local pkg="$1" dest="$2" ext="tgz"
    echo "Downloading $pkg ..."
    curl -fL --retry 3 -o "$TMP/ms.$ext" "$MS_BASE/$pkg.$ext"
    mkdir -p "$TMP/ms"
    tar -xzf "$TMP/ms.$ext" -C "$TMP/ms"
    rm -rf "$dest" && mkdir -p "$dest"
    cp -a "$TMP/ms/$pkg/." "$dest/"
}

case "$BACKEND" in
cpu)
    fetch_microsoft "onnxruntime-${OS}-$([[ $OS == osx ]] && echo universal2 || echo "$ARCH")-${VERSION}" "$DEST"
    ;;
coreml)
    [[ $OS == osx ]] || { echo "coreml is only available on macOS" >&2; exit 1; }
    fetch_microsoft "onnxruntime-osx-universal2-${VERSION}" "$DEST"
    ;;
cuda)
    [[ $OS == linux && $ARCH == x64 ]] || { echo "cuda build is only fetched for Linux x64" >&2; exit 1; }
    fetch_microsoft "onnxruntime-linux-x64-gpu-${VERSION}" "$DEST"
    ;;
openvino)
    [[ $OS == linux && $ARCH == x64 ]] || { echo "openvino build is only fetched for Linux x64" >&2; exit 1; }
    command -v pip3 >/dev/null || command -v pip >/dev/null || { echo "pip is required" >&2; exit 1; }
    PIP="$(command -v pip3 || command -v pip)"
    echo "Downloading onnxruntime-openvino ${VERSION} wheel ..."
    "$PIP" download --quiet --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 \
        --python-version 3.12 "onnxruntime-openvino==${VERSION}" -d "$TMP/wheel"
    # A wheel is a zip file
    (cd "$TMP/wheel" && python3 -c 'import sys,zipfile,glob; zipfile.ZipFile(glob.glob("*.whl")[0]).extractall("x")' 2>/dev/null \
        || unzip -q ./*.whl -d x)
    # Headers + the unversioned lib name come from the matching Microsoft CPU release
    fetch_microsoft "onnxruntime-linux-x64-${VERSION}" "$TMP/headers"
    rm -rf "$DEST" && mkdir -p "$DEST/lib"
    cp -a "$TMP/headers/include" "$DEST/include"
    cp -a "$TMP/wheel/x/onnxruntime/capi/"*.so* "$DEST/lib/"
    # CMake links libonnxruntime.so and the loader asks for its soname (libonnxruntime.so.1);
    # the wheel only ships the fully versioned file
    ln -sf "libonnxruntime.so.${VERSION}" "$DEST/lib/libonnxruntime.so.${VERSION%%.*}"
    ln -sf "libonnxruntime.so.${VERSION}" "$DEST/lib/libonnxruntime.so"
    ;;
*)
    echo "Unknown backend '$BACKEND' (cpu | cuda | openvino | coreml)" >&2
    usage
    ;;
esac

echo "ONNX Runtime ($BACKEND, $VERSION) ready: $DEST"
echo "Use it with:  cmake -S . -B build -DONNXRUNTIME_DIR=$DEST"
