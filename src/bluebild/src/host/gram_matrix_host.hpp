#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto gram_matrix_host(ContextInternal& ctx, std::size_t m, std::size_t n, const std::complex<T>* w,
                      std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl, std::complex<T>* g,
                      std::size_t ldg) -> void;

}  // namespace bluebild
