#include "host/intensity_field_data_host.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "host/eigensolver_host.hpp"
#include "host/gram_matrix_host.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
auto intensity_field_data_host(ContextInternal& ctx, T wl, std::size_t m, std::size_t n,
                               std::size_t nEig, const std::complex<T>* s, std::size_t lds,
                               const std::complex<T>* w, std::size_t ldw, const T* xyz,
                               std::size_t ldxyz, T* d, std::complex<T>* v, std::size_t ldv,
                               std::size_t nCluster, const T* cluster, int* clusterIndices)
    -> void {
  auto bufferG = create_buffer<std::complex<T>>(ctx.allocators().host(), n * n);

  gram_matrix_host<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, bufferG.get(), n);

  int nEigOut = 0;
  eigh_host<T>(ctx, n, nEig, s, lds, bufferG.get(), n, &nEigOut, d, v, ldv);

  if (nEigOut) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(nEigOut); ++i) {
      T min = std::numeric_limits<T>::max();
      int idx = 0;
      for (std::size_t j = 0; j < nCluster; ++j) {
        auto diff = std::abs(cluster[j] - d[i]);
        if(diff < min) {
          min = diff;
          idx = static_cast<int>(j);
        }
      }
      clusterIndices[i] = idx;
    }
  }
}

template auto intensity_field_data_host<float>(
    ContextInternal& ctx, float wl, std::size_t m, std::size_t n, std::size_t nEig,
    const std::complex<float>* s, std::size_t lds, const std::complex<float>* w, std::size_t ldw,
    const float* xyz, std::size_t ldxyz, float* d, std::complex<float>* v, std::size_t ldv,
    std::size_t nCluster, const float* cluster, int* clusterIndices) -> void;

template auto intensity_field_data_host<double>(
    ContextInternal& ctx, double wl, std::size_t m, std::size_t n, std::size_t nEig,
    const std::complex<double>* s, std::size_t lds, const std::complex<double>* w, std::size_t ldw,
    const double* xyz, std::size_t ldxyz, double* d, std::complex<double>* v, std::size_t ldv,
    std::size_t nCluster, const double* cluster, int* clusterIndices) -> void;

}  // namespace bluebild
