#pragma once

#include <cstddef>

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{

template <typename T>
auto reverse_1_gpu(gpu::StreamType stream, std::size_t n, T *x) -> void;

template <typename T>
auto reverse_2_gpu(gpu::StreamType stream, std::size_t m, std::size_t n, T *x,
                   std::size_t ld) -> void;
}

