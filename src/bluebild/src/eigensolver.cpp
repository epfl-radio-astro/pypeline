#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "bluebild/context.hpp"
#include "context_internal.hpp"
#include "host/eigensolver_host.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "memory/buffer.hpp"
#endif

namespace bluebild {
template <typename T, typename>
BLUEBILD_EXPORT auto eigh(ContextInternal &ctx, int m, int nEig,
                          const std::complex<T> *a, int lda,
                          const std::complex<T> *b, int ldb, int *nEigOut, T *d,
                          std::complex<T> *v, int ldv) -> void {
  if (ctx.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        // Syncronize with default stream. TODO: replace with event
        gpu::check_status(gpu::stream_synchronize(nullptr));

        BufferType<gpu::ComplexType<T>> aBuffer, bBuffer, vBuffer;
        BufferType<T> dBuffer;
        auto aDevice = reinterpret_cast<const gpu::ComplexType<T>*>(a);
        auto bDevice = reinterpret_cast<const gpu::ComplexType<T>*>(b);
        auto dDevice = d;
        auto vDevice = reinterpret_cast<gpu::ComplexType<T>*>(v);
        int ldaDevice = lda;
        int ldbDevice = ldb;
        int ldvDevice = ldv;

        // copy input if required
        if(!is_device_ptr(a)) {
          aBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), m * m);
          ldaDevice = m;
          aDevice = aBuffer.get();
          gpu::check_status(gpu::memcpy_2d_async(
              aBuffer.get(), ldaDevice * sizeof(gpu::ComplexType<T>), a,
              lda * sizeof(gpu::ComplexType<T>), m * sizeof(gpu::ComplexType<T>), m,
              gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
        }

        if(b && !is_device_ptr(b)) {
          bBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), m * m);
          ldbDevice = m;
          bDevice = bBuffer.get();
          gpu::check_status(gpu::memcpy_2d_async(
              bBuffer.get(), ldbDevice * sizeof(gpu::ComplexType<T>), b,
              ldb * sizeof(gpu::ComplexType<T>), m * sizeof(gpu::ComplexType<T>), m,
              gpu::flag::MemcpyHostToDevice, ctx.gpu_stream()));
        }

        if(!is_device_ptr(v)) {
          vBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), nEig * m);
          ldvDevice = m;
          vDevice = vBuffer.get();
        }

        if(!is_device_ptr(d)) {
          dBuffer = create_buffer<T>(ctx.allocators().gpu(), nEig);
          dDevice = dBuffer.get();
        }

        // call eigh on GPU
        eigh_gpu<T>(ctx, m, nEig, aDevice, ldaDevice, bDevice, ldbDevice, nEigOut,
                                   dDevice, vDevice, ldvDevice);

        // copy back results if required
        if(vBuffer) {
          gpu::check_status(gpu::memcpy_2d_async(
              v, ldv * sizeof(gpu::ComplexType<T>), vBuffer.get(),
              ldvDevice * sizeof(gpu::ComplexType<T>), m * sizeof(gpu::ComplexType<T>), nEig,
              gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
        }
        if(dBuffer) {
          gpu::check_status(gpu::memcpy_async(d, dDevice, nEig * sizeof(T),
                                              gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
        }


        // syncronize with stream to be synchronous with host
        gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream()));

#else
        throw GPUSupportError();
#endif
      } else {
        eigh_host<T>(ctx, m, nEig, a, lda, b, ldb, nEigOut, d, v, ldv);
      }
}

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_eigh_s(BluebildContext ctx, int m,
                                              int nEig, const void *a, int lda,
                                              const void *b, int ldb,
                                              int *nEigOut, float *d, void *v,
                                              int ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    eigh<float>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), m,
        nEig, reinterpret_cast<const std::complex<float> *>(a), lda,
        reinterpret_cast<const std::complex<float> *>(b), ldb, nEigOut, d,
        reinterpret_cast<std::complex<float> *>(v), ldv);
  } catch (const bluebild::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_eigh_d(BluebildContext ctx, int m,
                                              int nEig, const void *a, int lda,
                                              const void *b, int ldb,
                                              int *nEigOut, double *d, void *v,
                                              int ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    eigh<double>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), m,
        nEig, reinterpret_cast<const std::complex<double> *>(a), lda,
        reinterpret_cast<const std::complex<double> *>(b), ldb, nEigOut, d,
        reinterpret_cast<std::complex<double> *>(v), ldv);
  } catch (const bluebild::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}

} // namespace bluebild
