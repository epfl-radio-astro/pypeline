#pragma once

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template <typename T>
auto reverse_1_gpu(gpu::StreamType stream, int n, T* x) -> void;

template <typename T>
auto reverse_2_gpu(gpu::StreamType stream, int m, int n, T* x, int ld) -> void;

}

