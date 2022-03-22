#pragma once

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
template <typename T>
auto inv_square_1d_gpu(gpu::StreamType stream, int n, T* x) -> void;
}
