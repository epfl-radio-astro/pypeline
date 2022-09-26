#pragma once

#include <complex>
#include <type_traits>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/context.hpp"
#include "bluebild/nufft_3d3.hpp"

enum BluebildFilter {
  BLUEBILD_FILTER_LSQ,
  BLUEBILD_FILTER_STD,
  BLUEBILD_FILTER_SQRT,
  BLUEBILD_FILTER_INV
};

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

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto eigh(ContextInternal &ctx, int m, int nEig,
                          const std::complex<T> *a, int lda,
                          const std::complex<T> *b, int ldb, int *nEigOut, T *d,
                          std::complex<T> *v, int ldv) -> void;

template <typename T, typename = std::enable_if_t<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, float>>>
BLUEBILD_EXPORT auto
virtual_vis(Context &ctx, int nFilter, const BluebildFilter *filter,
            int nIntervals, const T *intervals, int ldIntervals, int nEig,
            const T *D, int nAntenna, const std::complex<T> *V, int ldv,
            int nBeam, const std::complex<T> *W, int ldw,
            std::complex<T> *virtVis, int ldVirtVis1, int ldVirtVis2,
            int ldVirtVis3) -> void;
} // namespace bluebild
