
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "gpu/kernels/reverse.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/solver_api.hpp"

namespace bluebild {

template <typename T>
auto eigh_gpu(ContextInternal& ctx, int m, int nEig, const gpu::ComplexType<T>* a, int lda,
              const gpu::ComplexType<T>* b, int ldb, int* nEigOut, T* d, gpu::ComplexType<T>* v,
              int ldv) -> void {
  // TODO: add fill mode
  using ComplexType = gpu::ComplexType<T>;
  using ScalarType = T;

  auto aBuffer = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);  // Matrix A
  auto dBuffer = create_buffer<T>(ctx.allocators().gpu(), m);  // Matrix A

  gpu::check_status(gpu::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a,
                                         lda * sizeof(ComplexType), m * sizeof(ComplexType), m,
                                         gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
  int hMeig = 0;

  // compute positive eigenvalues
  gpu::eigensolver::solve(ctx, 'V', 'V', 'L', m, aBuffer.get(), m,
                          std::numeric_limits<T>::epsilon(),
                          std::numeric_limits<T>::max(), 1, m, &hMeig,
                          dBuffer.get());

  if (b) {
    auto bBuffer = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);  // Matrix B
    gpu::check_status(gpu::memcpy_2d_async(bBuffer.get(), m * sizeof(ComplexType), b,
                                           ldb * sizeof(ComplexType), m * sizeof(ComplexType), m,
                                           gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
    if (hMeig != m) {
      // reconstuct 'a' without negative eigenvalues (v * diag(d) * v^H)
      auto dComplexD = create_buffer<ComplexType>(ctx.allocators().gpu(), m);
      auto cD = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);
      auto newABuffer = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);

      // copy scalar eigenvalues to complex for multiplication
      gpu::check_status(
          gpu::memset_async(dComplexD.get(), 0, hMeig * sizeof(ComplexType), ctx.gpu_stream()));
      gpu::check_status(gpu::memcpy_2d_async(dComplexD.get(), sizeof(ComplexType), dBuffer.get(),
                                             sizeof(ScalarType), sizeof(ScalarType), hMeig,
                                             gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));

      gpu::blas::check_status(gpu::blas::dgmm(ctx.gpu_blas_handle(), gpu::blas::side::right, m,
                                              hMeig, aBuffer.get(), m, dComplexD.get(), 1, cD.get(), m));
      ComplexType alpha{1, 0};
      ComplexType beta{0, 0};
      gpu::blas::check_status(gpu::blas::gemm(ctx.gpu_blas_handle(), gpu::blas::operation::None,
                                              gpu::blas::operation::ConjugateTranspose, m, m, hMeig,
                                              &alpha, cD.get(), m, aBuffer.get(), m, &beta,
                                              newABuffer.get(), m));
      std::swap(newABuffer, aBuffer);
    } else {
      // a was overwritten by eigensolver
      gpu::check_status(gpu::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a,
                                             lda * sizeof(ComplexType), m * sizeof(ComplexType), m,
                                             gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
    }

    // compute positive general eigenvalues
    gpu::eigensolver::solve(ctx, 'V', 'V', 'L', m, aBuffer.get(), m,
                            bBuffer.get(), m, std::numeric_limits<T>::epsilon(),
                            std::numeric_limits<T>::max(), 1, m, &hMeig,
                            dBuffer.get());
  }

  if (hMeig > 1) {
    // reverse order, such that large eigenvalues are first
    reverse_1_gpu<ScalarType>(ctx.gpu_stream(), hMeig, dBuffer.get());
    reverse_2_gpu(ctx.gpu_stream(), m, hMeig, aBuffer.get(), m);
  }

  if (hMeig < nEig) {
    // fewer positive eigenvalues found than requested. Setting others to 0.
    gpu::check_status(
        gpu::memset_async(d + hMeig, 0, (nEig - hMeig) * sizeof(ScalarType), ctx.gpu_stream()));
    gpu::check_status(gpu::memset_async(
        aBuffer.get() + hMeig * m, 0, (nEig - hMeig) * m * sizeof(ComplexType), ctx.gpu_stream()));
  }

  // copy results to output
  gpu::check_status(gpu::memcpy_async(d, dBuffer.get(), nEig * sizeof(ScalarType),
                                      gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::memcpy_2d_async(v, ldv * sizeof(ComplexType), aBuffer.get(),
                                         m * sizeof(ComplexType), m * sizeof(ComplexType), nEig,
                                         gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));

  *nEigOut = std::min<int>(hMeig, nEig);
}

template auto eigh_gpu<float>(ContextInternal& ctx, int m, int nEig,
                              const gpu::ComplexType<float>* a, int lda,
                              const gpu::ComplexType<float>* b, int ldb, int* nEigOut, float* d,
                              gpu::ComplexType<float>* v, int ldv) -> void;

template auto eigh_gpu<double>(ContextInternal& ctx, int m, int nEig,
                               const gpu::ComplexType<double>* a, int lda,
                               const gpu::ComplexType<double>* b, int ldb, int* nEigOut, double* d,
                               gpu::ComplexType<double>* v, int ldv) -> void;

}  // namespace bluebild
