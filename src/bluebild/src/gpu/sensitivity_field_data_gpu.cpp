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
auto sensitivity_field_data_gpu(ContextInternal &ctx, T wl, std::size_t m,
                                std::size_t n, std::size_t nEig,
                                const gpu::ComplexType<T> *w, std::size_t ldw,
                                const T *xyz, std::size_t ldxyz, T *d,
                                gpu::ComplexType<T> *v, std::size_t ldv)
    -> void {
  auto gD = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), n * n);

  gram_matrix_gpu<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, gD.get(), n);

  std::size_t nEigOut = 0;
  eigh_gpu<T>(ctx, n, nEig, gD.get(), n, nullptr, 0, &nEigOut, d, v, ldv);

  if (nEigOut)
    inv_square_1d_gpu(ctx.gpu_stream(), nEigOut, d);
}

template auto sensitivity_field_data_gpu<float>(
    ContextInternal &ctx, float wl, std::size_t m, std::size_t n,
    std::size_t nEig, const gpu::ComplexType<float> *w, std::size_t ldw,
    const float *xyz, std::size_t ldxyz, float *d, gpu::ComplexType<float> *v,
    std::size_t ldv) -> void;

template auto sensitivity_field_data_gpu<double>(
    ContextInternal &ctx, double wl, std::size_t m, std::size_t n,
    std::size_t nEig, const gpu::ComplexType<double> *w, std::size_t ldw,
    const double *xyz, std::size_t ldxyz, double *d,
    gpu::ComplexType<double> *v, std::size_t ldv) -> void;
} // namespace bluebild
