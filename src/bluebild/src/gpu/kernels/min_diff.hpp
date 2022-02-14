#pragma once

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
template <typename T>
auto min_diff_1d_gpu(gpu::StreamType stream, int m, const T* x, int n, const T* y, int* indices)
    -> void;
}
