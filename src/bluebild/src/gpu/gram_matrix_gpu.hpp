#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {
template <typename T>
auto gram_matrix_gpu(ContextInternal& ctx, int m, int n, const gpu::ComplexType<T>* w, int ldw,
                     const T* xyz, int ldxyz, T wl, gpu::ComplexType<T>* g, int ldg) -> void;
}  // namespace bluebild
