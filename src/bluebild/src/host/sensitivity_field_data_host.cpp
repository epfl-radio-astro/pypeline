#include "host/sensitivity_field_data_host.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

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
auto sensitivity_field_data_host(ContextInternal& ctx, T wl, std::size_t m, std::size_t n,
                                 std::size_t nEig, const std::complex<T>* w, std::size_t ldw,
                                 const T* xyz, std::size_t ldxyz, T* d, std::complex<T>* v,
                                 std::size_t ldv) -> void {

  auto bufferG = create_buffer<std::complex<T>>(ctx.allocators().host(), n * n);

  gram_matrix_host<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, bufferG.get(), n);

  int nEigOut = 0;
  eigh_host<T>(ctx, n, nEig, bufferG.get(), n, nullptr, 0, &nEigOut, d, v, ldv);

  if (nEigOut) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(nEigOut); ++i) {
      d[i] = 1 / (d[i] * d[i]);
    }
  }
}

template auto sensitivity_field_data_host<float>(ContextInternal& ctx, float wl, std::size_t m,
                                                 std::size_t n, std::size_t nEig,
                                                 const std::complex<float>* w, std::size_t ldw,
                                                 const float* xyz, std::size_t ldxyz, float* d,
                                                 std::complex<float>* v, std::size_t ldv) -> void;

template auto sensitivity_field_data_host<double>(ContextInternal& ctx, double wl, std::size_t m,
                                                  std::size_t n, std::size_t nEig,
                                                  const std::complex<double>* w, std::size_t ldw,
                                                  const double* xyz, std::size_t ldxyz, double* d,
                                                  std::complex<double>* v, std::size_t ldv) -> void;
}  // namespace bluebild
