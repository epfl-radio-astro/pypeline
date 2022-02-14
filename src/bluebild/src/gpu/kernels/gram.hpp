#pragma once

#include <complex>
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
template <typename T>
auto gram_gpu(gpu::StreamType stream, int n, const T* x, const T* y, const T* z, T wl,
              gpu::ComplexType<T>* g, int ldg) -> void;
}

