#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "gpu/gram_matrix_gpu.hpp"
#include "memory/buffer.hpp"
#include "gpu/kernels/inv_square.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
auto sensitivity_field_data_gpu(ContextInternal &ctx, T wl, int m, int n,
                                int nEig, const gpu::ComplexType<T> *w, int ldw,
                                const T *xyz, int ldxyz, T *d,
                                gpu::ComplexType<T> *v, int ldv) -> void {
  using ScalarType = T;

  auto gD = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), n * n);

  gram_matrix_gpu<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, gD.get(), n);

  int nEigOut = 0;
  eigh_gpu<T>(ctx, n, nEig, gD.get(), n, nullptr, 0, &nEigOut, d, v, ldv);

  if (nEigOut)
    inv_square_1d_gpu(ctx.gpu_stream(), nEigOut, d);
}

template auto sensitivity_field_data_gpu<float>(
    ContextInternal &ctx, float wl, int m, int n, int nEig,
    const gpu::ComplexType<float> *w, int ldw, const float *xyz, int ldxyz,
    float *d, gpu::ComplexType<float> *v, int ldv) -> void;

template auto sensitivity_field_data_gpu<double>(
    ContextInternal &ctx, double wl, int m, int n, int nEig,
    const gpu::ComplexType<double> *w, int ldw, const double *xyz, int ldxyz,
    double *d, gpu::ComplexType<double> *v, int ldv) -> void;
} // namespace bluebild
