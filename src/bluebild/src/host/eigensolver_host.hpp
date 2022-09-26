#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto eigh_host(ContextInternal& ctx, std::size_t m, std::size_t nEig, const std::complex<T>* a,
               std::size_t lda, const std::complex<T>* b, std::size_t ldb, int* nEigOut, T* d,
               std::complex<T>* v, std::size_t ldv) -> void;

}  // namespace bluebild
