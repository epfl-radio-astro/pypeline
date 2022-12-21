#pragma once

#include <cstddef>

#include <complex>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
template <typename T>
auto gram_gpu(gpu::StreamType stream, std::size_t n, const T *x, const T *y,
              const T *z, T wl, gpu::ComplexType<T> *g, std::size_t ldg)
    -> void;
}

