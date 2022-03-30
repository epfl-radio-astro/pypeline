#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "bluebild/context.hpp"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "host/sensitivity_field_data_host.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#include "gpu/sensitivity_field_data_gpu.hpp"
#endif

namespace bluebild {

template <typename T>
auto sensitivity_field_data(ContextInternal& ctx, T wl, int m, int n, int nEig,
                            const std::complex<T>* w, int ldw, const T* xyz, int ldxyz, T* d,
                            std::complex<T>* v, int ldv) -> void {
  if (ctx.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
        // Syncronize with default stream. TODO: replace with event
        gpu::stream_synchronize(nullptr);

        BufferType<gpu::ComplexType<T>> wBuffer, vBuffer;
        BufferType<T> dBuffer, xyzBuffer;
        auto wDevice = reinterpret_cast<const gpu::ComplexType<T>*>(w);
        auto vDevice = reinterpret_cast<gpu::ComplexType<T>*>(v);
        auto xyzDevice = xyz;
        auto dDevice = d;
        int ldwDevice = ldw;
        int ldvDevice = ldv;
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

        // prepare output
        if(!is_device_ptr(v)) {
          vBuffer = create_buffer<gpu::ComplexType<T>>(ctx.allocators().gpu(), nEig * n);
          ldvDevice = n;
          vDevice = vBuffer.get();
        }
        if(!is_device_ptr(d)) {
          dBuffer = create_buffer<T>(ctx.allocators().gpu(), nEig);
          dDevice = dBuffer.get();
        }


        // call intensity_field_data on gpu
        auto devInfo =
            sensitivity_field_data_gpu<T>(ctx, wl, m, n, nEig, wDevice, ldwDevice, xyzDevice,
                                          ldxyzDevice, dDevice, vDevice, ldvDevice);

        if(dBuffer) {
          gpu::check_status(gpu::memcpy_async(d, dDevice, nEig * sizeof(T),
                                              gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
        }
        if(vBuffer) {
          gpu::check_status(gpu::memcpy_2d_async(
              v, ldv * sizeof(gpu::ComplexType<T>), vBuffer.get(),
              ldvDevice * sizeof(gpu::ComplexType<T>), n * sizeof(gpu::ComplexType<T>), nEig,
              gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));
        }

        int hostInfo[2];
        gpu::check_status(gpu::memcpy_async(hostInfo, devInfo.get(), 2 * sizeof(int),
                                            gpu::flag::MemcpyDeviceToHost, ctx.gpu_stream()));

        // syncronize with stream to be synchronous with host
        gpu::check_status(gpu::stream_synchronize(ctx.gpu_stream())); 

        // check if eigensolver threw error
        if (hostInfo[0] || hostInfo[1]) {
          throw EigensolverError();
        }
#else
        throw GPUSupportError();
#endif
  } else {
    sensitivity_field_data_host<T>(ctx, wl, m, n, nEig, w, ldw, xyz, ldxyz, d, v, ldv);
  }
}

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_sensitivity_field_data_s(BluebildContext ctx, float wl,
                                                                int m, int n, int nEig, void* w,
                                                                int ldw, const float* xyz,
                                                                int ldxyz, float* d, void* v,
                                                                int ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    sensitivity_field_data<float>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), wl, m,
        n, nEig, reinterpret_cast<const std::complex<float> *>(w), ldw, xyz,
        ldxyz, d, reinterpret_cast<std::complex<float> *>(v), ldv);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_sensitivity_field_data_d(BluebildContext ctx, double wl,
                                                                int m, int n, int nEig, void* w,
                                                                int ldw, const double* xyz,
                                                                int ldxyz, double* d, void* v,
                                                                int ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    sensitivity_field_data<double>(
        *InternalContextAccessor::get(*reinterpret_cast<Context *>(ctx)), wl, m,
        n, nEig, reinterpret_cast<const std::complex<double> *>(w), ldw, xyz,
        ldxyz, d, reinterpret_cast<std::complex<double> *>(v), ldv);
  } catch (const bluebild::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}

}  // namespace bluebild
