#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {
template <typename T>
auto gram_matrix_gpu(ContextInternal& ctx, int m, int n, const std::complex<T>* w, int ldw,
                     const T* xyz, int ldxyz, T wl, std::complex<T>* g, int ldg) -> void;
}  // namespace bluebild
