#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "gpu/kernels/gram.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

template <typename T>
auto gram_matrix_gpu(ContextInternal& ctx, int m, int n, const gpu::ComplexType<T>* w, int ldw,
                     const T* xyz, int ldxyz, T wl, gpu::ComplexType<T>* g, int ldg) -> void {
  using ComplexType = gpu::ComplexType<T>;

  // Syncronize with default stream. TODO: replace with event
  gpu::stream_synchronize(nullptr);

  auto baseD = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);

  {
    auto xyzD = create_buffer<T>(ctx.allocators().gpu(), 3 * m);
    gpu::check_status(gpu::memcpy_2d_async(xyzD.get(), m * sizeof(T), xyz, ldxyz * sizeof(T),
                                           m * sizeof(T), 3, gpu::flag::MemcpyDefault,
                                           ctx.gpu_stream()));
    gram_gpu(ctx.gpu_stream(), m, xyzD.get(), xyzD.get() + m, xyzD.get() + 2 * m, wl, baseD.get(),
             m);
  }

  auto wD = create_buffer<ComplexType>(ctx.allocators().gpu(), m * n);
  auto cD = create_buffer<ComplexType>(ctx.allocators().gpu(), m * n);
  gpu::check_status(gpu::memcpy_2d_async(wD.get(), m * sizeof(ComplexType), w,
                                         ldw * sizeof(ComplexType), m * sizeof(ComplexType), n,
                                         gpu::flag::MemcpyDefault, ctx.gpu_stream()));
  ComplexType alpha{1, 0};
  ComplexType beta{0, 0};
  gpu::blas::check_status(gpu::blas::symm(ctx.gpu_blas_handle(), gpu::blas::side::left,
                                          gpu::blas::fill::lower, m, n, &alpha, baseD.get(), m,
                                          wD.get(), m, &beta, cD.get(), m));
  auto gD = create_buffer<ComplexType>(ctx.allocators().gpu(), n * n);
  gpu::blas::check_status(gpu::blas::gemm(
      ctx.gpu_blas_handle(), gpu::blas::operation::ConjugateTranspose, gpu::blas::operation::None,
      n, n, m, &alpha, wD.get(), m, cD.get(), m, &beta, gD.get(), n));

  gpu::check_status(gpu::memcpy_2d_async(g, ldg * sizeof(ComplexType), gD.get(),
                                         n * sizeof(ComplexType), n * sizeof(ComplexType), n,
                                         gpu::flag::MemcpyDefault, ctx.gpu_stream()));

  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));  // syncronize with stream
}

template auto gram_matrix_gpu<float>(ContextInternal& ctx, int m, int n,
                                     const gpu::ComplexType<float>* w, int ldw, const float* xyz,
                                     int ldxyz, float wl, gpu::ComplexType<float>* g, int ldg)
    -> void;

template auto gram_matrix_gpu<double>(ContextInternal& ctx, int m, int n,
                                      const gpu::ComplexType<double>* w, int ldw, const double* xyz,
                                      int ldxyz, double wl, gpu::ComplexType<double>* g, int ldg)
    -> void;

}  // namespace bluebild
