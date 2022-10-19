#!/bin/bash

usage() { echo "Usage $0 [--no-config]" 1>&2; exit 1; }

config_cmake=true

while getopts ":h-:" optchar; do
    case "$optchar" in
        -)  case "$OPTARG" in
                no-config) config_cmake=false ;;
            esac;;
        *) echo "config"; usage ;;
    esac
done

THIS_DIR=$(dirname $(realpath $0))

set -ex

build_dir="build_rocm"
config="RelWithDebInfo"

rocm_home="/opt/rocm-5.2.0"
rocm_version="5.2.1"

LLVM_PATH=/usr/local/llvm-git
export HIP_CLANG_PATH=${LLVM_PATH}/bin
export LIBRARY_PATH=${LLVM_PATH}/lib/clang/13.0.0/lib/linux

if $config_cmake; then
    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so
    # rm -fr ${THIS_DIR}/${build_dir}/${config}/build/lib

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            CMAKE_HIP_COMPILER=${LLVM_PATH}/bin/clang++ \
            CMAKE_HIP_FLAGS="-DCK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=1" \
            CMAKE_EXPORT_COMPILE_COMMANDS=ON \
            onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        --use_rocm \
        --rocm_version=5.2 \
        --rocm_home /opt/rocm --nccl_home=/opt/rocm \
        --enable_rocm_profiling \
        --enable_training \
        --enable_training_torch_interop \
        --build_wheel \
        --skip_submodule_sync --skip_tests

        # --use_migraphx \

    ${THIS_DIR}/create-fake-dist-info.sh ${THIS_DIR}/${build_dir}/${config}/build/lib/
else
    printf "\n\tSkipping config\n\n"
    cmake --build ${THIS_DIR}/${build_dir}/${config}
fi
