#pragma once

#include <complex>
#include <type_traits>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/context.hpp"
#include "bluebild/nufft_3d3.hpp"

namespace bluebild {

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
gram_matrix(Context &ctx, int m, int n, const std::complex<T> *w, int ldw,
            const T *xyz, int ldxyz, T wl, std::complex<T> *g, int ldg) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto sensitivity_field_data(Context &ctx, T wl, int m, int n,
                                            int nEig, const std::complex<T> *w,
                                            int ldw, const T *xyz, int ldxyz,
                                            T *d, std::complex<T> *v, int ldv)
    -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
intensity_field_data(Context &ctx, T wl, int m, int n, int nEig,
                     const std::complex<T> *s, int lds,
                     const std::complex<T> *w, int ldw, const T *xyz, int ldxyz,
                     T *d, std::complex<T> *v, int ldv, int nCluster,
                     const T *cluster, int *clusterIndices) -> void;

} // namespace bluebild
