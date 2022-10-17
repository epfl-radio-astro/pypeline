#pragma once

#include <complex>
#include <type_traits>
#include <cstddef>

#include "bluebild/config.h"
#include "bluebild/enums.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/context.hpp"
#include "bluebild/nufft_3d3.hpp"
#include "bluebild/nufft_synthesis.hpp"
#include "bluebild/standard_synthesis.hpp"

namespace bluebild {

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto gram_matrix(Context &ctx, std::size_t m, std::size_t n,
                                 const std::complex<T> *w, std::size_t ldw,
                                 const T *xyz, std::size_t ldxyz, T wl,
                                 std::complex<T> *g, std::size_t ldg) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
sensitivity_field_data(Context &ctx, T wl, std::size_t m, std::size_t n,
                       std::size_t nEig, const std::complex<T> *w,
                       std::size_t ldw, const T *xyz, std::size_t ldxyz, T *d,
                       std::complex<T> *v, std::size_t ldv) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
intensity_field_data(Context &ctx, T wl, std::size_t m, std::size_t n,
                     std::size_t nEig, const std::complex<T> *s,
                     std::size_t lds, const std::complex<T> *w, std::size_t ldw,
                     const T *xyz, std::size_t ldxyz, T *d, std::complex<T> *v,
                     std::size_t ldv) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto eigh(ContextInternal &ctx, std::size_t m, std::size_t nEig,
                          const std::complex<T> *a, std::size_t lda,
                          const std::complex<T> *b, std::size_t ldb,
                          std::size_t *nEigOut, T *d, std::complex<T> *v,
                          std::size_t ldv) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
virtual_vis(Context &ctx, std::size_t nFilter, const BluebildFilter *filter,
            std::size_t nIntervals, const T *intervals, std::size_t ldIntervals,
            std::size_t nEig, const T *D, std::size_t nAntenna,
            const std::complex<T> *V, std::size_t ldv, std::size_t nBeam,
            const std::complex<T> *W, std::size_t ldw, std::complex<T> *virtVis,
            std::size_t ldVirtVis1, std::size_t ldVirtVis2,
            std::size_t ldVirtVis3) -> void;
} // namespace bluebild
