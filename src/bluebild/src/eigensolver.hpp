#pragma once

#include <complex>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {
template <typename T>
auto eigh_gpu(ContextInternal& ctx, int m, int nEig, const std::complex<T>* a, int lda,
              const std::complex<T>* b, int ldb, int* nEigOut, T* d, std::complex<T>* v, int ldv)
    -> void;
}  // namespace bluebild
