#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bluebild {
template <typename T>
auto eigh_gpu(ContextInternal& ctx, int m, int nEig, const gpu::ComplexType<T>* a, int lda,
              const gpu::ComplexType<T>* b, int ldb, int* nEigOut, T* d, gpu::ComplexType<T>* v,
              int ldv) -> BufferType<int>;
}  // namespace bluebild
