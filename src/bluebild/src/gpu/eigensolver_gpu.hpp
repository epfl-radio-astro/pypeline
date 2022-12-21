#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bluebild {
template <typename T>
auto eigh_gpu(ContextInternal &ctx, std::size_t m, std::size_t nEig,
              const gpu::ComplexType<T> *a, std::size_t lda,
              const gpu::ComplexType<T> *b, std::size_t ldb,
              std::size_t *nEigOut, T *d, gpu::ComplexType<T> *v,
              std::size_t ldv) -> void;
}  // namespace bluebild
