#pragma once

#include <cstddef>

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild{
template <typename T>
auto gemmexp_gpu(gpu::StreamType stream, std::size_t nEig, std::size_t nPixel,
                 std::size_t nAntenna, T alpha,
                 const gpu::ComplexType<T> *vUnbeam, std::size_t ldv,
                 const T *xyz, std::size_t ldxyz, const T *pixelX,
                 const T *pixelY, const T *pixelZ, T *out, std::size_t ldout)
    -> void;
} // namespace bluebild
