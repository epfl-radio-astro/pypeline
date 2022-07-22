
#include <cusolverDn.h>

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

namespace eigensolver {
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il,
                   int iu) -> int {
  int lwork = 0;
  if (cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il,
                   int iu) -> int {
  int lwork = 0;
  if (cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, cuComplex* A, int lda, float vl, float vu, int il,
                   int iu) -> int {
  int lwork = 0;
  if (cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double vl, double vu,
                   int il, int iu) -> int {
  int lwork = 0;
  if (cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il, int iu,
           int* hMeig, float* W, float* work, int lwork, int* devInfo) -> void {
  if (cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, hMeig, W, work, lwork,
                        devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il, int iu,
           int* hMeig, double* W, double* work, int lwork, int* devInfo) -> void {
  if (cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, hMeig, W, work, lwork,
                        devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, cuComplex* A, int lda, float vl, float vu, int il, int iu,
           int* hMeig, float* W, cuComplex* work, int lwork, int* devInfo) -> void {
  if (cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, hMeig, W, work, lwork,
                        devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double vl, double vu, int il,
           int iu, int* hMeig, double* W, cuDoubleComplex* work, int lwork, int* devInfo) -> void {
  if (cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, hMeig, W, work, lwork,
                        devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

}  // namespace eigensolver

namespace general_eigensolver {
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl,
                   float vu, int il, int iu) -> int {
  int lwork = 0;
  if (cusolverDnSsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl,
                   double vu, int il, int iu) -> int {
  int lwork = 0;
  if (cusolverDnDsygvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb,
                   float vl, float vu, int il, int iu) -> int {
  int lwork = 0;
  if (cusolverDnChegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}
auto worspace_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
                   cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B,
                   int ldb, double vl, double vu, int il, int iu) -> int {
  int lwork = 0;
  if (cusolverDnZhegvdx_bufferSize(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
  return lwork;
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl, float vu,
           int il, int iu, int* hMeig, float* W, float* work, int lwork, int* devInfo) -> void {
  if (cusolverDnSsygvdx(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                        il, iu, hMeig, W, work, lwork, devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl,
           double vu, int il, int iu, int* hMeig, double* W, double* work, int lwork, int* devInfo)
    -> void {
  if (cusolverDnDsygvdx(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                        il, iu, hMeig, W, work, lwork, devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, cuComplex* A, int lda, cuComplex* B, int ldb, float vl,
           float vu, int il, int iu, int* hMeig, float* W, cuComplex* work, int lwork, int* devInfo)
    -> void {
  if (cusolverDnChegvdx(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                        il, iu, hMeig, W, work, lwork, devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

auto solve(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range,
           cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb,
           double vl, double vu, int il, int iu, int* hMeig, double* W, cuDoubleComplex* work,
           int lwork, int* devInfo) -> void {
  if (cusolverDnZhegvdx(handle, CUSOLVER_EIG_TYPE_1, jobz, range, uplo, n, A, lda, B, ldb, vl, vu,
                        il, iu, hMeig, W, work, lwork, devInfo) != CUSOLVER_STATUS_SUCCESS)
    throw GPUError();
}

}  // namespace general_eigensolver

template <typename T>
auto eigh_gpu(ContextInternal& ctx, int m, int nEig, const gpu::ComplexType<T>* a, int lda,
              const gpu::ComplexType<T>* b, int ldb, int* nEigOut, T* d, gpu::ComplexType<T>* v,
              int ldv) -> BufferType<int> {
  // TODO: add fill mode
  using ComplexType = gpu::ComplexType<T>;
  using ScalarType = T;

  auto aBuffer = create_buffer<ComplexType>(ctx.allocators().gpu(), m * m);  // Matrix A
  auto dBuffer = create_buffer<T>(ctx.allocators().gpu(), m);  // Matrix A

  gpu::check_status(gpu::memcpy_2d_async(aBuffer.get(), m * sizeof(ComplexType), a,
                                         lda * sizeof(ComplexType), m * sizeof(ComplexType), m,
                                         gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
  int hMeig = 0;

  int lwork = eigensolver::worspace_size(ctx.gpu_solver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                         CUSOLVER_EIG_RANGE_V, CUBLAS_FILL_MODE_LOWER, m, aBuffer.get(),
                                         m, std::numeric_limits<T>::epsilon(),
                                         std::numeric_limits<T>::max(), 1, m);
  auto workspace = create_buffer<ComplexType>(ctx.allocators().gpu(), lwork);
  auto devInfo = create_buffer<int>(ctx.allocators().gpu(), 2);
  // make sure info is always 0. Second entry might not be set otherwise.
  gpu::memset_async(devInfo.get(), 0, 2 * sizeof(int), ctx.gpu_stream());

  // compute positive eigenvalues
  eigensolver::solve(ctx.gpu_solver_handle(), CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_V,
                     CUBLAS_FILL_MODE_LOWER, m, aBuffer.get(), m, std::numeric_limits<T>::epsilon(),
                     std::numeric_limits<T>::max(), 1, m, &hMeig, dBuffer.get(), workspace.get(),
                     lwork, devInfo.get());

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

    // allocate new work buffer
    lwork = general_eigensolver::worspace_size(
        ctx.gpu_solver_handle(), CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_V,
        CUBLAS_FILL_MODE_LOWER, m, aBuffer.get(), m, bBuffer.get(), m,
        std::numeric_limits<T>::epsilon(), std::numeric_limits<T>::max(), 1, m);
    workspace.reset();
    workspace = create_buffer<ComplexType>(ctx.allocators().gpu(), lwork);

    // compute positive general eigenvalues
    general_eigensolver::solve(ctx.gpu_solver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                               CUSOLVER_EIG_RANGE_V, CUBLAS_FILL_MODE_LOWER, m, aBuffer.get(), m,
                               bBuffer.get(), m, std::numeric_limits<T>::epsilon(),
                               std::numeric_limits<T>::max(), 1, m, &hMeig, dBuffer.get(),
                               workspace.get(), lwork, devInfo.get() + 1);
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
  return devInfo;
}

template auto eigh_gpu<float>(ContextInternal& ctx, int m, int nEig,
                              const gpu::ComplexType<float>* a, int lda,
                              const gpu::ComplexType<float>* b, int ldb, int* nEigOut, float* d,
                              gpu::ComplexType<float>* v, int ldv) -> BufferType<int>;

template auto eigh_gpu<double>(ContextInternal& ctx, int m, int nEig,
                               const gpu::ComplexType<double>* a, int lda,
                               const gpu::ComplexType<double>* b, int ldb, int* nEigOut, double* d,
                               gpu::ComplexType<double>* v, int ldv) -> BufferType<int>;

}  // namespace bluebild
