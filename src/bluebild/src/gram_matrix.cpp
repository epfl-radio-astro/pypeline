#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "bluebild/context.hpp"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "host/gram_matrix_host.hpp"


#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "eigensolver.hpp"
#include "gpu/gram_matrix_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T>
auto gram_matrix(ContextInternal& ctx, int m, int n, const std::complex<T>* w, int ldw, const T* xyz,
                     int ldxyz, T wl, std::complex<T>* g, int ldg) -> void {
      if(ctx.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        // Syncronize with default stream. TODO: replace with event
        gpu::stream_synchronize(nullptr);

        BufferType<gpu::ComplexType<T>> wBuffer, gBuffer;
        BufferType<T> xyzBuffer;
        auto wDevice = reinterpret_cast<const gpu::ComplexType<T>*>(w);
        auto gDevice = reinterpret_cast<gpu::ComplexType<T>*>(g);
        auto xyzDevice = xyz;
        int ldwDevice = ldw;
        int ldgDevice = ldg;
        int ldxyzDevice = ldxyz;


        // copy input if required
        if(!is_device_ptr(w)) {
          wBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), m * n);
          ldwDevice = m;
          wDevice = wBuffer.get();
          gpu::check_status(gpu::memcpy_2d_async(
              wBuffer.get(), m * sizeof(gpu::ComplexType<T>), w, ldw * sizeof(gpu::ComplexType<T>),
              m * sizeof(gpu::ComplexType<T>), n, gpu::flag::MemcpyDefault, ctx.gpu_stream()));
        }

        if(!is_device_ptr(xyz)) {
          xyzBuffer = create_buffer<T>(ctx.allocators().gpu(), 3 * m);
          ldxyzDevice = m;
          xyzDevice = xyzBuffer.get();
          gpu::check_status(gpu::memcpy_2d_async(xyzBuffer.get(), m * sizeof(T), xyz,
                                                 ldxyz * sizeof(T), m * sizeof(T), 3,
                                                 gpu::flag::MemcpyDefault, ctx.gpu_stream()));
        }
        if(!is_device_ptr(g)) {
          gBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), n * n);
          ldgDevice = n;
          gDevice = gBuffer.get();
        }

        // call gram on gpu
        gram_matrix_gpu<T>(ctx, m, n, wDevice, ldwDevice, xyzDevice, ldxyzDevice, wl, gDevice,
                           ldgDevice);

        // copy back results if required
        if(gBuffer) {
          gpu::check_status(gpu::memcpy_2d_async(
              g, ldg * sizeof(gpu::ComplexType<T>), gBuffer.get(), n * sizeof(gpu::ComplexType<T>),
              n * sizeof(gpu::ComplexType<T>), n, gpu::flag::MemcpyDefault, ctx.gpu_stream()));
        }

        // syncronize with stream to be synchronous with host
        gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream())); 
#else
        throw GPUSupportError();
#endif
      } else {
        gram_matrix_host<T>(ctx, m, n, w, ldw, xyz, ldxyz, wl, g, ldg);
      }
}

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_s(BluebildContext ctx, int m, int n,
                                                     const void* w, int ldw, const float* xyz,
                                                     int ldxyz, float wl, void* g, int ldg) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<float>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), m, n,
        reinterpret_cast<const std::complex<float> *>(w), ldw, xyz, ldxyz, wl,
        reinterpret_cast<std::complex<float> *>(g), ldg);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_d(BluebildContext ctx, int m, int n,
                                                     const void* w, int ldw, const double* xyz,
                                                     int ldxyz, double wl, void* g, int ldg) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    gram_matrix<double>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), m, n,
        reinterpret_cast<const std::complex<double> *>(w), ldw, xyz, ldxyz, wl,
        reinterpret_cast<std::complex<double> *>(g), ldg);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}

}  // namespace bluebild
