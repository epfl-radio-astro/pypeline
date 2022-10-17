#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"

namespace bluebild {

template <typename T>
auto intensity_field_data_host(ContextInternal &ctx, T wl, std::size_t m,
                               std::size_t n, std::size_t nEig,
                               const std::complex<T> *s, std::size_t lds,
                               const std::complex<T> *w, std::size_t ldw,
                               const T *xyz, std::size_t ldxyz, T *d,
                               std::complex<T> *v, std::size_t ldv) -> void;

}  // namespace bluebild
