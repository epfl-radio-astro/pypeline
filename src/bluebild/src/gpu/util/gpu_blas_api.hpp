#pragma once

#include <stdexcept>
#include <utility>

#include "bluebild/config.h"

#if defined(BLUEBILD_CUDA)
#include <cublas_v2.h>

#elif defined(BLUEBILD_ROCM)
#include <rocblas.h>

#else
#error Either BLUEBILD_CUDA or BLUEBILD_ROCM must be defined!
#endif

#include "bluebild/exceptions.hpp"

namespace bluebild {
namespace gpu {
namespace blas {

#if defined(BLUEBILD_CUDA)
using HandleType = cublasHandle_t;
using StatusType = cublasStatus_t;
using OperationType = cublasOperation_t;
using SideModeType = cublasSideMode_t;
using FillModeType = cublasFillMode_t;
#endif

#if defined(BLUEBILD_ROCM)
using HandleType = rocblas_handle;
using StatusType = rocblas_status;
using OperationType = rocblas_operation;
using SideModeType = rocblas_side;
using FillModeType = rocblas_fill;
#endif

namespace operation {
#if defined(BLUEBILD_CUDA)
constexpr auto None = CUBLAS_OP_N;
constexpr auto Transpose = CUBLAS_OP_T;
constexpr auto ConjugateTranspose = CUBLAS_OP_C;
#endif

#if defined(BLUEBILD_ROCM)
constexpr auto None = rocblas_operation_none;
constexpr auto Transpose = rocblas_operation_transpose;
constexpr auto ConjugateTranspose = rocblas_operation_conjugate_transpose;
#endif
}  // namespace operation

namespace side {
#if defined(BLUEBILD_CUDA)
constexpr auto left = CUBLAS_SIDE_LEFT;
constexpr auto right = CUBLAS_SIDE_RIGHT;
#endif

#if defined(BLUEBILD_ROCM)
constexpr auto left = rocblas_side_left;
constexpr auto right = rocblas_side_right;
#endif
}  // namespace side

namespace fill {
#if defined(BLUEBILD_CUDA)
constexpr auto upper = CUBLAS_FILL_MODE_UPPER;
constexpr auto lower = CUBLAS_FILL_MODE_LOWER;
constexpr auto full = CUBLAS_FILL_MODE_FULL;
#endif

#if defined(BLUEBILD_ROCM)
constexpr auto upper = rocblas_fill_upper;
constexpr auto lower = rocblas_fill_lower;
constexpr auto full = rocblas_fill_full;
#endif
}  // namespace side

namespace status {
#if defined(BLUEBILD_CUDA)
constexpr auto Success = CUBLAS_STATUS_SUCCESS;
#endif

#if defined(BLUEBILD_ROCM)
constexpr auto Success = rocblas_status_success;
#endif

static const char *get_string(StatusType error) {
#if defined(BLUEBILD_CUDA)
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_ERROR";
  }
#endif

#if defined(BLUEBILD_ROCM)
  switch (error) {
    case rocblas_status_success:
      return "rocblas_status_success";

    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";

    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";

    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";

    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";

    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";

    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";

    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";

    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";

    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";

    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
    default:
      return "rocblas_error";
  }
#endif

  return "<unknown>";
}
}  // namespace status

inline auto check_status(StatusType error) -> void {
  if (error != status::Success) {
    throw GPUBlasError();
  }
}

// =======================================
// Forwarding functions of to GPU BLAS API
// =======================================
template <typename... ARGS>
inline auto create(ARGS &&...args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasCreate(std::forward<ARGS>(args)...);
#else
  return rocblas_create_handle(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto destroy(ARGS &&...args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasDestroy(std::forward<ARGS>(args)...);
#else
  return rocblas_destroy_handle(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto set_stream(ARGS &&...args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasSetStream(std::forward<ARGS>(args)...);
#else
  return rocblas_set_stream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto get_stream(ARGS &&...args) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasGetStream(std::forward<ARGS>(args)...);
#else
  return rocblas_get_stream(std::forward<ARGS>(args)...);
#endif
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const float *alpha, const float *A, int lda, const float *B, int ldb,
                 const float *beta, float *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const double *alpha, const double *A, int lda, const double *B, int ldb,
                 const double *beta, double *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const ComplexFloatType *alpha, const ComplexFloatType *A, int lda,
                 const ComplexFloatType *B, int ldb, const ComplexFloatType *beta,
                 ComplexFloatType *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_cgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const rocblas_float_complex *>(alpha),
                       reinterpret_cast<const rocblas_float_complex *>(A), lda,
                       reinterpret_cast<const rocblas_float_complex *>(B), ldb,
                       reinterpret_cast<const rocblas_float_complex *>(beta),
                       reinterpret_cast<rocblas_float_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

inline auto gemm(HandleType handle, OperationType transa, OperationType transb, int m, int n, int k,
                 const ComplexDoubleType *alpha, const ComplexDoubleType *A, int lda,
                 const ComplexDoubleType *B, int ldb, const ComplexDoubleType *beta,
                 ComplexDoubleType *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_zgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const rocblas_double_complex *>(alpha),
                       reinterpret_cast<const rocblas_double_complex *>(A), lda,
                       reinterpret_cast<const rocblas_double_complex *>(B), ldb,
                       reinterpret_cast<const rocblas_double_complex *>(beta),
                       reinterpret_cast<rocblas_double_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

inline auto gemm_batched(HandleType handle,
                         OperationType transa,
                         OperationType transb,
                         int m, int n, int k,
                         const ComplexFloatType *alpha,
                         const ComplexFloatType* const A[], int lda,
                         const ComplexFloatType* const B[], int ldb,
                         const ComplexFloatType *beta,
                         ComplexFloatType* const C[], int ldc,
                         int batchCount
                         ) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#else
  return rocblas_cgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif  // BLUEBILD_CUDA
}

inline auto gemm_batched(HandleType handle,
                         OperationType transa,
                         OperationType transb,
                         int m, int n, int k,
                         const ComplexDoubleType *alpha,
                         const ComplexDoubleType* const A[], int lda,
                         const ComplexDoubleType* const B[], int ldb,
                         const ComplexDoubleType *beta,
                         ComplexDoubleType* const C[], int ldc,
                         int batchCount
                         ) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#else
  return rocblas_zgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
#endif  // BLUEBILD_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const float *A, int lda,
                 const float *x, int incx, float *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
  return rocblas_sdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const double *A, int lda,
                 const double *x, int incx, double *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
  return rocblas_ddgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const ComplexFloatType *A,
                 int lda, const ComplexFloatType *x, int incx, ComplexFloatType *C, int ldc)
    -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
  return rocblas_cdgmm(handle, mode, m, n,
                       reinterpret_cast<const rocblas_float_complex *>(A), lda,
                       reinterpret_cast<const rocblas_float_complex *>(x), incx,
                       reinterpret_cast<rocblas_float_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

inline auto dgmm(HandleType handle, SideModeType mode, int m, int n, const ComplexDoubleType *A,
                 int lda, const ComplexDoubleType *x, int incx, ComplexDoubleType *C, int ldc)
    -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
#else
  return rocblas_zdgmm(
      handle, mode, m, n, reinterpret_cast<const rocblas_double_complex *>(A),
      lda, reinterpret_cast<const rocblas_double_complex *>(x), incx,
      reinterpret_cast<rocblas_double_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const float *alpha, const float *A, int lda, const float *B, int ldb,
                 const float *beta, float *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_ssymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const double *alpha, const double *A, int lda, const double *B, int ldb,
                 const double *beta, double *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_dsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#endif  // BLUEBILD_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const ComplexFloatType *alpha, const ComplexFloatType *A, int lda,
                 const ComplexFloatType *B, int ldb, const ComplexFloatType *beta,
                 ComplexFloatType *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_csymm(handle, side, uplo, m, n,
                       reinterpret_cast<const rocblas_float_complex *>(alpha),
                       reinterpret_cast<const rocblas_float_complex *>(A), lda,
                       reinterpret_cast<const rocblas_float_complex *>(B), ldb,
                       reinterpret_cast<const rocblas_float_complex *>(beta),
                       reinterpret_cast<rocblas_float_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

inline auto symm(HandleType handle, SideModeType side, FillModeType uplo, int m, int n,
                 const ComplexDoubleType *alpha, const ComplexDoubleType *A, int lda,
                 const ComplexDoubleType *B, int ldb, const ComplexDoubleType *beta,
                 ComplexDoubleType *C, int ldc) -> StatusType {
#if defined(BLUEBILD_CUDA)
  return cublasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  return rocblas_zsymm(handle, side, uplo, m, n,
                       reinterpret_cast<const rocblas_double_complex *>(alpha),
                       reinterpret_cast<const rocblas_double_complex *>(A), lda,
                       reinterpret_cast<const rocblas_double_complex *>(B), ldb,
                       reinterpret_cast<const rocblas_double_complex *>(beta),
                       reinterpret_cast<rocblas_double_complex *>(C), ldc);
#endif  // BLUEBILD_CUDA
}

}  // namespace blas
}  // namespace gpu
}  // namespace bluebild
