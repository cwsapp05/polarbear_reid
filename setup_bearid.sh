#!/usr/bin/env bash
# Build the BearID command-line tools (bearface, bearchip, bearembed) and fetch their networks.
# Tested with dlib 19.24.6 + Boost 1.83 on Ubuntu 24.04 (CPU build). For a GPU build, install
# CUDA + cuDNN first; dlib's CMake turns CUDA on automatically when it finds them.
#
# System packages (Ubuntu/Debian):
#   sudo apt-get install build-essential cmake git libboost-filesystem-dev libboost-system-dev \
#        libopenblas-dev liblapack-dev libjpeg-dev libpng-dev libx11-dev
# libx11-dev is required: bearchip includes dlib's GUI headers even though it never opens a window.
set -euo pipefail
cd "$(dirname "$0")"
case "$(pwd)" in /mnt/*) echo "WARNING: building under /mnt fails on WSL (git chmod error). Copy this script to ~/pbreid-build and run it there; then copy tools/ back." >&2 ;; esac
mkdir -p tools && cd tools

[ -d dlib ] || git clone --depth 1 --branch v19.24.6 https://github.com/davisking/dlib.git
[ -d bearid-dlib ] || git clone --depth 1 https://github.com/bearid/bearid-dlib.git
[ -d bearid-models ] || git clone --depth 1 https://github.com/hypraptive/bearid-models.git

cd bearid-dlib
# Boost >= 1.89 no longer ships a boost_system library (it is header-only), so asking CMake for
# the "system" component fails and BearID silently skips its targets. filesystem alone suffices.
sed -i 's/find_package(Boost COMPONENTS filesystem system)/find_package(Boost COMPONENTS filesystem)/' CMakeLists.txt
mkdir -p build && cd build
cmake -DDLIB_PATH="$(cd ../../dlib && pwd)" -DCMAKE_BUILD_TYPE=Release .. | tee cmake.log
grep -iE "DLIB WILL USE CUDA|DLIB WILL NOT USE CUDA|Disabling CUDA" cmake.log || true
cmake --build . --config Release -j"$(nproc)"
ls -l bearface bearchip bearembed
echo "BearID tools built in $(pwd)"
