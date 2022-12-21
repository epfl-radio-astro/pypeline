#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
template <typename T>
auto gram_matrix_gpu(ContextInternal &ctx, std::size_t m, std::size_t n,
                     const gpu::ComplexType<T> *w, std::size_t ldw,
                     const T *xyz, std::size_t ldxyz, T wl,
                     gpu::ComplexType<T> *g, std::size_t ldg) -> void;
}  // namespace bluebild
