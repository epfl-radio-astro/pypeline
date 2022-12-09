#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto gemmexp(const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const T           alpha,
             const T* __restrict__ A,
             const std::size_t lda,
             const T* __restrict__ B,
             const std::size_t ldb,
             std::complex<T>* __restrict__ C,
             const std::size_t ldc) -> void;

}  // namespace bluebild
