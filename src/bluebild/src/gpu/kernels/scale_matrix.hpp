#pragma once

#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
namespace gpu {
template <typename T>
auto scale_matrix(gpu::StreamType stream, int m, int n,
                  const gpu::ComplexType<T> *A, int lda, const T *x,
                  gpu::ComplexType<T> *B, int ldb) -> void;
}
}
