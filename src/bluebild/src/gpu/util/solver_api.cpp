#include "gpu/util/solver_api.hpp"
#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "host/lapack_api.hpp"
#include "memory/buffer.hpp"
#include "context_internal.hpp"

#ifdef BLUEBILD_MAGMA
#include <magma.h>
#include <magma_c.h>
#include <magma_types.h>
#include <magma_z.h>
#else
#include <cusolverDn.h>
#endif

namespace bluebild {
namespace gpu {
namespace eigensolver {

namespace {
#ifdef BLUEBILD_MAGMA
struct MagmaInit {
  MagmaInit() { magma_init(); }
  MagmaInit(const MagmaInit &) = delete;
  ~MagmaInit() { magma_finalize(); }
};

MagmaInit MAGMA_INIT_GUARD;
#endif

auto convert_jobz(char j) {
#ifdef BLUEBILD_MAGMA
  switch (j) {
  case 'N':
  case 'n':
    return MagmaNoVec;
  case 'V':
  case 'v':
    return MagmaVec;
  }
  throw InternalError();
  return MagmaVec;
#else
  switch (j) {
  case 'N':
  case 'n':
    return CUSOLVER_EIG_MODE_NOVECTOR;
  case 'V':
  case 'v':
    return CUSOLVER_EIG_MODE_VECTOR;
  }
  throw InternalError();
  return CUSOLVER_EIG_MODE_VECTOR;
#endif
}

auto convert_range(char r) {
#ifdef BLUEBILD_MAGMA
  switch (r) {
  case 'A':
  case 'a':
    return MagmaRangeAll;
  case 'V':
  case 'v':
    return MagmaRangeV;
  case 'I':
  case 'i':
    return MagmaRangeI;
  }
  throw InternalError();
  return MagmaRangeAll;
#else
  switch (r) {
  case 'A':
  case 'a':
    return CUSOLVER_EIG_RANGE_ALL;
  case 'V':
  case 'v':
    return CUSOLVER_EIG_RANGE_V;
  case 'I':
  case 'i':
    return CUSOLVER_EIG_RANGE_I;
  }
  throw InternalError();
  return CUSOLVER_EIG_RANGE_ALL;
#endif
}

auto convert_uplo(char u) {
#ifdef BLUEBILD_MAGMA
  switch (u) {
  case 'L':
  case 'l':
    return MagmaLower;
  case 'U':
  case 'u':
    return MagmaUpper;
  }
  throw InternalError();
  return MagmaLower;
#else
  switch (u) {
  case 'L':
  case 'l':
    return CUBLAS_FILL_MODE_LOWER;
  case 'U':
  case 'u':
    return CUBLAS_FILL_MODE_UPPER;
  }
  throw InternalError();
  return CUBLAS_FILL_MODE_LOWER;
#endif
}

} // namespace

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<float> *a, int lda, float vl, float vu, int il,
           int iu, int *m, float *w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BLUEBILD_MAGMA
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  const float abstol = 2 * lapack::slamch('S');

  auto z = create_buffer<ComplexType<float>>(ctx.allocators().gpu(), n * n);
  int ldz = n;

  auto wHost = create_buffer<float>(ctx.allocators().pinned(), n);
  auto wA = create_buffer<ComplexType<float>>(ctx.allocators().host(), n * n);
  auto wZ = create_buffer<ComplexType<float>>(ctx.allocators().host(), n * n);
  auto rwork = create_buffer<float>(ctx.allocators().host(), 7 * n);
  auto iwork = create_buffer<int>(ctx.allocators().host(), 5 * n);
  auto ifail = create_buffer<int>(ctx.allocators().host(), n);
  int info = 0;
  gpu::ComplexType<float> worksize;
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n,
                   reinterpret_cast<magmaFloatComplex *>(a), lda, vl, vu, il,
                   iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaFloatComplex *>(z.get()), ldz,
                   reinterpret_cast<magmaFloatComplex *>(wA.get()), n,
                   reinterpret_cast<magmaFloatComplex *>(wZ.get()), n,
                   reinterpret_cast<magmaFloatComplex *>(&worksize), -1,
                   rwork.get(), iwork.get(), ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n)
    lwork = 2 * n;
  auto work = create_buffer<ComplexType<float>>(ctx.allocators().host(), lwork);
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n,
                   reinterpret_cast<magmaFloatComplex *>(a), lda, vl, vu, il,
                   iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaFloatComplex *>(z.get()), ldz,
                   reinterpret_cast<magmaFloatComplex *>(wA.get()), n,
                   reinterpret_cast<magmaFloatComplex *>(wZ.get()), n,
                   reinterpret_cast<magmaFloatComplex *>(work.get()), lwork,
                   rwork.get(), iwork.get(), ifail.get(), &info);

  if (info != 0)
    throw EigensolverError();

  gpu::check_status(gpu::memcpy_async(w, wHost.get(), (*m) * sizeof(float),
                                      gpu::flag::MemcpyHostToDevice,
                                      ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(a, lda * sizeof(gpu::ComplexType<float>), z.get(),
                           ldz * sizeof(gpu::ComplexType<float>),
                           n * sizeof(gpu::ComplexType<float>), *m,
                           gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

#else

  int lwork = 0;
  if (cusolverDnCheevdx_bufferSize(ctx.gpu_solver_handle(), jobzEnum, rangeEnum,
                                   uploEnum, n, a, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace =
      create_buffer<ComplexType<float>>(ctx.allocators().gpu(), lwork);
  auto devInfo = create_buffer<int>(ctx.allocators().gpu(), 1);
  if (cusolverDnCheevdx(ctx.gpu_solver_handle(), jobzEnum, rangeEnum, uploEnum,
                        n, a, lda, vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  gpu::check_status(gpu::memcpy_async(&hostInfo, devInfo.get(), sizeof(int),
                                      gpu::flag::MemcpyDeviceToHost,
                                      ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<double> *a, int lda, double vl, double vu, int il,
           int iu, int *m, double *w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BLUEBILD_MAGMA
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  const double abstol = 2 * lapack::dlamch('S');

  auto z = create_buffer<ComplexType<double>>(ctx.allocators().gpu(), n * n);
  int ldz = n;

  auto wHost = create_buffer<double>(ctx.allocators().pinned(), n);
  auto wA = create_buffer<ComplexType<double>>(ctx.allocators().host(), n * n);
  auto wZ = create_buffer<ComplexType<double>>(ctx.allocators().host(), n * n);
  auto rwork = create_buffer<double>(ctx.allocators().host(), 7 * n);
  auto iwork = create_buffer<int>(ctx.allocators().host(), 5 * n);
  auto ifail = create_buffer<int>(ctx.allocators().host(), n);
  int info = 0;
  gpu::ComplexType<double> worksize;
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n,
                   reinterpret_cast<magmaDoubleComplex *>(a), lda, vl, vu, il,
                   iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaDoubleComplex *>(z.get()), ldz,
                   reinterpret_cast<magmaDoubleComplex *>(wA.get()), n,
                   reinterpret_cast<magmaDoubleComplex *>(wZ.get()), n,
                   reinterpret_cast<magmaDoubleComplex *>(&worksize), -1,
                   rwork.get(), iwork.get(), ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n)
    lwork = 2 * n;
  auto work =
      create_buffer<ComplexType<double>>(ctx.allocators().host(), lwork);
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n,
                   reinterpret_cast<magmaDoubleComplex *>(a), lda, vl, vu, il,
                   iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaDoubleComplex *>(z.get()), ldz,
                   reinterpret_cast<magmaDoubleComplex *>(wA.get()), n,
                   reinterpret_cast<magmaDoubleComplex *>(wZ.get()), n,
                   reinterpret_cast<magmaDoubleComplex *>(work.get()), lwork,
                   rwork.get(), iwork.get(), ifail.get(), &info);

  if (info != 0)
    throw EigensolverError();

  gpu::check_status(gpu::memcpy_async(w, wHost.get(), (*m) * sizeof(double),
                                      gpu::flag::MemcpyHostToDevice,
                                      ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(a, lda * sizeof(gpu::ComplexType<double>), z.get(),
                           ldz * sizeof(gpu::ComplexType<double>),
                           n * sizeof(gpu::ComplexType<double>), *m,
                           gpu::flag::MemcpyDeviceToDevice, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

#else

  int lwork = 0;
  if (cusolverDnZheevdx_bufferSize(ctx.gpu_solver_handle(), jobzEnum, rangeEnum,
                                   uploEnum, n, a, lda, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace =
      create_buffer<ComplexType<double>>(ctx.allocators().gpu(), lwork);
  auto devInfo = create_buffer<int>(ctx.allocators().gpu(), 1);
  // make sure info is always 0. Second entry might not be set otherwise.
  gpu::memset_async(devInfo.get(), 0, sizeof(int), ctx.gpu_stream());
  if (cusolverDnZheevdx(ctx.gpu_solver_handle(), jobzEnum, rangeEnum, uploEnum,
                        n, a, lda, vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  gpu::check_status(gpu::memcpy_async(&hostInfo, devInfo.get(), sizeof(int),
                                      gpu::flag::MemcpyDeviceToHost,
                                      ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<float> *a, int lda, gpu::ComplexType<float> *b,
           int ldb, float vl, float vu, int il, int iu, int *m, float *w)
    -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BLUEBILD_MAGMA
  auto aHost =
      create_buffer<ComplexType<float>>(ctx.allocators().pinned(), n * n);
  auto bHost =
      create_buffer<ComplexType<float>>(ctx.allocators().pinned(), n * n);
  auto zHost =
      create_buffer<ComplexType<float>>(ctx.allocators().pinned(), n * n);
  gpu::check_status(
      gpu::memcpy_2d_async(aHost.get(), n * sizeof(gpu::ComplexType<float>), a,
                           lda * sizeof(gpu::ComplexType<float>),
                           n * sizeof(gpu::ComplexType<float>), n,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(bHost.get(), n * sizeof(gpu::ComplexType<float>), b,
                           ldb * sizeof(gpu::ComplexType<float>),
                           n * sizeof(gpu::ComplexType<float>), n,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

  const float abstol = 2 * lapack::slamch('S');

  int ldz = n;

  auto wHost = create_buffer<float>(ctx.allocators().pinned(), n);
  auto rwork = create_buffer<float>(ctx.allocators().host(), 7 * n);
  auto iwork = create_buffer<int>(ctx.allocators().host(), 5 * n);
  auto ifail = create_buffer<int>(ctx.allocators().host(), n);
  int info = 0;
  gpu::ComplexType<float> worksize;
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex *>(aHost.get()), n,
               reinterpret_cast<magmaFloatComplex *>(bHost.get()), n, vl, vu,
               il, iu, abstol, m, wHost.get(),
               reinterpret_cast<magmaFloatComplex *>(zHost.get()), ldz,
               reinterpret_cast<magmaFloatComplex *>(&worksize), -1,
               rwork.get(), iwork.get(), ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n)
    lwork = 2 * n;
  auto work = create_buffer<ComplexType<float>>(ctx.allocators().host(), lwork);
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex *>(aHost.get()), n,
               reinterpret_cast<magmaFloatComplex *>(bHost.get()), n, vl, vu,
               il, iu, abstol, m, wHost.get(),
               reinterpret_cast<magmaFloatComplex *>(zHost.get()), ldz,
               reinterpret_cast<magmaFloatComplex *>(work.get()), lwork,
               rwork.get(), iwork.get(), ifail.get(), &info);

  if (info != 0)
    throw EigensolverError();

  gpu::check_status(gpu::memcpy_async(w, wHost.get(), (*m) * sizeof(float),
                                      gpu::flag::MemcpyHostToDevice,
                                      ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(a, lda * sizeof(gpu::ComplexType<float>),
                           zHost.get(), ldz * sizeof(gpu::ComplexType<float>),
                           n * sizeof(gpu::ComplexType<float>), *m,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

#else

  int lwork = 0;
  if (cusolverDnChegvdx_bufferSize(ctx.gpu_solver_handle(), CUSOLVER_EIG_TYPE_1,
                                   jobzEnum, rangeEnum, uploEnum, n, a, lda, b,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace =
      create_buffer<ComplexType<float>>(ctx.allocators().gpu(), lwork);
  auto devInfo = create_buffer<int>(ctx.allocators().gpu(), 2);
  if (cusolverDnChegvdx(ctx.gpu_solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum,
                        rangeEnum, uploEnum, n, a, lda, b, ldb, vl, vu, il, iu,
                        m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  gpu::check_status(gpu::memcpy_async(&hostInfo, devInfo.get(), sizeof(int),
                                      gpu::flag::MemcpyDeviceToHost,
                                      ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<double> *a, int lda, gpu::ComplexType<double> *b,
           int ldb, double vl, double vu, int il, int iu, int *m, double *w)
    -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BLUEBILD_MAGMA
  auto aHost =
      create_buffer<ComplexType<double>>(ctx.allocators().pinned(), n * n);
  auto bHost =
      create_buffer<ComplexType<double>>(ctx.allocators().pinned(), n * n);
  auto zHost =
      create_buffer<ComplexType<double>>(ctx.allocators().pinned(), n * n);
  gpu::check_status(
      gpu::memcpy_2d_async(aHost.get(), n * sizeof(gpu::ComplexType<double>), a,
                           lda * sizeof(gpu::ComplexType<double>),
                           n * sizeof(gpu::ComplexType<double>), n,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(bHost.get(), n * sizeof(gpu::ComplexType<double>), b,
                           ldb * sizeof(gpu::ComplexType<double>),
                           n * sizeof(gpu::ComplexType<double>), n,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

  const double abstol = 2 * lapack::dlamch('S');

  int ldz = n;

  auto wHost = create_buffer<double>(ctx.allocators().pinned(), n);
  auto rwork = create_buffer<double>(ctx.allocators().host(), 7 * n);
  auto iwork = create_buffer<int>(ctx.allocators().host(), 5 * n);
  auto ifail = create_buffer<int>(ctx.allocators().host(), n);
  int info = 0;
  gpu::ComplexType<double> worksize;
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex *>(aHost.get()), n,
               reinterpret_cast<magmaDoubleComplex *>(bHost.get()), n, vl, vu,
               il, iu, abstol, m, wHost.get(),
               reinterpret_cast<magmaDoubleComplex *>(zHost.get()), ldz,
               reinterpret_cast<magmaDoubleComplex *>(&worksize), -1,
               rwork.get(), iwork.get(), ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n)
    lwork = 2 * n;
  auto work =
      create_buffer<ComplexType<double>>(ctx.allocators().host(), lwork);
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex *>(aHost.get()), n,
               reinterpret_cast<magmaDoubleComplex *>(bHost.get()), n, vl, vu,
               il, iu, abstol, m, wHost.get(),
               reinterpret_cast<magmaDoubleComplex *>(zHost.get()), ldz,
               reinterpret_cast<magmaDoubleComplex *>(work.get()), lwork,
               rwork.get(), iwork.get(), ifail.get(), &info);

  if (info != 0)
    throw EigensolverError();

  gpu::check_status(gpu::memcpy_async(w, wHost.get(), (*m) * sizeof(double),
                                      gpu::flag::MemcpyHostToDevice,
                                      ctx.gpu_stream()));
  gpu::check_status(
      gpu::memcpy_2d_async(a, lda * sizeof(gpu::ComplexType<double>),
                           zHost.get(), ldz * sizeof(gpu::ComplexType<double>),
                           n * sizeof(gpu::ComplexType<double>), *m,
                           gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
#else
  int lwork = 0;
  if (cusolverDnZhegvdx_bufferSize(ctx.gpu_solver_handle(), CUSOLVER_EIG_TYPE_1,
                                   jobzEnum, rangeEnum, uploEnum, n, a, lda, b,
                                   ldb, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace =
      create_buffer<ComplexType<double>>(ctx.allocators().gpu(), lwork);
  auto devInfo = create_buffer<int>(ctx.allocators().gpu(), 1);
  if (cusolverDnZhegvdx(ctx.gpu_solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum,
                        rangeEnum, uploEnum, n, a, lda, b, ldb, vl, vu, il, iu,
                        m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  gpu::check_status(gpu::memcpy_async(&hostInfo, devInfo.get(), sizeof(int),
                                      gpu::flag::MemcpyDeviceToHost,
                                      ctx.gpu_stream()));
  gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}
} // namespace eigensolver

} // namespace gpu
} // namespace bluebild
