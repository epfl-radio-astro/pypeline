#pragma once

#include <cstddef>

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {
template <typename T>
auto scale_matrix(gpu::StreamType stream, std::size_t m, std::size_t n,
                  const gpu::ComplexType<T> *A, std::size_t lda, const T *x,
                  gpu::ComplexType<T> *B, std::size_t ldb) -> void;
}
}
