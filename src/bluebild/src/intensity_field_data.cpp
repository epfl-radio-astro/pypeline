#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "bluebild/context.hpp"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "host/intensity_field_data_host.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "gpu/intensity_field_data_gpu.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T, typename>
BLUEBILD_EXPORT auto
intensity_field_data(Context &ctx, T wl, std::size_t m, std::size_t n,
                     std::size_t nEig, const std::complex<T> *s,
                     std::size_t lds, const std::complex<T> *w, std::size_t ldw,
                     const T *xyz, std::size_t ldxyz, T *d, std::complex<T> *v,
                     std::size_t ldv) -> void {

  auto &ctxInternal = *InternalContextAccessor::get(ctx);
  if (ctxInternal.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
    // Syncronize with default stream. TODO: replace with event
    gpu::check_status(gpu::stream_synchronize(nullptr));

    BufferType<gpu::ComplexType<T>> sBuffer, wBuffer, vBuffer;
    BufferType<T> dBuffer, xyzBuffer;
    auto wDevice = reinterpret_cast<const gpu::ComplexType<T> *>(w);
    auto sDevice = reinterpret_cast<const gpu::ComplexType<T> *>(s);
    auto vDevice = reinterpret_cast<gpu::ComplexType<T> *>(v);
    auto xyzDevice = xyz;
    auto dDevice = d;
    std::size_t ldwDevice = ldw;
    std::size_t ldsDevice = lds;
    std::size_t ldvDevice = ldv;
    std::size_t ldxyzDevice = ldxyz;

    // copy input if required
    if (!is_device_ptr(w)) {
      wBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(), m * n);
      ldwDevice = m;
      wDevice = wBuffer.get();
      gpu::check_status(gpu::memcpy_2d_async(
          wBuffer.get(), m * sizeof(gpu::ComplexType<T>), w,
          ldw * sizeof(gpu::ComplexType<T>), m * sizeof(gpu::ComplexType<T>), n,
          gpu::flag::MemcpyDefault, ctxInternal.gpu_stream()));
    }
    if (!is_device_ptr(s)) {
      sBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(), n * n);
      ldsDevice = n;
      sDevice = sBuffer.get();
      gpu::check_status(gpu::memcpy_2d_async(
          sBuffer.get(), n * sizeof(gpu::ComplexType<T>), s,
          lds * sizeof(gpu::ComplexType<T>), n * sizeof(gpu::ComplexType<T>), n,
          gpu::flag::MemcpyDefault, ctxInternal.gpu_stream()));
    }

    if (!is_device_ptr(xyz)) {
      xyzBuffer = create_buffer<T>(ctxInternal.allocators().gpu(), 3 * m);
      ldxyzDevice = m;
      xyzDevice = xyzBuffer.get();
      gpu::check_status(gpu::memcpy_2d_async(
          xyzBuffer.get(), m * sizeof(T), xyz, ldxyz * sizeof(T), m * sizeof(T),
          3, gpu::flag::MemcpyDefault, ctxInternal.gpu_stream()));
    }

    // prepare output
    if (!is_device_ptr(v)) {
      vBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(), nEig * n);
      ldvDevice = n;
      vDevice = vBuffer.get();
    }
    if (!is_device_ptr(d)) {
      dBuffer = create_buffer<T>(ctxInternal.allocators().gpu(), nEig);
      dDevice = dBuffer.get();
    }

    // call intensity_field_data on gpu
    intensity_field_data_gpu<T>(ctxInternal, wl, m, n, nEig, sDevice, ldsDevice,
                                wDevice, ldwDevice, xyzDevice, ldxyzDevice,
                                dDevice, vDevice, ldvDevice);

    // copy back if required
    if (dBuffer) {
      gpu::check_status(gpu::memcpy_async(d, dDevice, nEig * sizeof(T),
                                          gpu::flag::MemcpyDeviceToHost,
                                          ctxInternal.gpu_stream()));
    }
    if (vBuffer) {
      gpu::check_status(gpu::memcpy_2d_async(
          v, ldv * sizeof(gpu::ComplexType<T>), vBuffer.get(),
          ldvDevice * sizeof(gpu::ComplexType<T>),
          n * sizeof(gpu::ComplexType<T>), nEig, gpu::flag::MemcpyDeviceToHost,
          ctxInternal.gpu_stream()));
    }

    // syncronize with stream to be synchronous with host
    gpu::check_status(gpu::stream_synchronize(ctxInternal.gpu_stream()));

#else
    throw GPUSupportError();
#endif
  } else {
    intensity_field_data_host<T>(ctxInternal, wl, m, n, nEig, s, lds, w, ldw,
                                 xyz, ldxyz, d, v, ldv);
  }
}

template auto
intensity_field_data(Context &ctx, float wl, std::size_t m, std::size_t n,
                     std::size_t nEig, const std::complex<float> *s,
                     std::size_t lds, const std::complex<float> *w,
                     std::size_t ldw, const float *xyz, std::size_t ldxyz,
                     float *d, std::complex<float> *v, std::size_t ldv) -> void;

template auto
intensity_field_data(Context &ctx, double wl, std::size_t m, std::size_t n,
                     std::size_t nEig, const std::complex<double> *s,
                     std::size_t lds, const std::complex<double> *w,
                     std::size_t ldw, const double *xyz, std::size_t ldxyz,
                     double *d, std::complex<double> *v, std::size_t ldv)
    -> void;

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_intensity_field_data_s(
    BluebildContext ctx, float wl, size_t m, size_t n, size_t nEig,
    const void *s, size_t lds, const void *w, size_t ldw, const float *xyz,
    size_t ldxyz, float *d, void *v, size_t ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    intensity_field_data<float>(
        *reinterpret_cast<Context *>(ctx), wl, m, n, nEig,
        reinterpret_cast<const std::complex<float> *>(s), lds,
        reinterpret_cast<const std::complex<float> *>(w), ldw, xyz, ldxyz, d,
        reinterpret_cast<std::complex<float> *>(v), ldv);
  } catch (const bluebild::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_intensity_field_data_d(
    BluebildContext ctx, double wl, size_t m, size_t n, size_t nEig,
    const void *s, size_t lds, const void *w, size_t ldw, const double *xyz,
    size_t ldxyz, double *d, void *v, size_t ldv) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    intensity_field_data<double>(
        *reinterpret_cast<Context *>(ctx), wl, m, n, nEig,
        reinterpret_cast<const std::complex<double> *>(s), lds,
        reinterpret_cast<const std::complex<double> *>(w), ldw, xyz, ldxyz, d,
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
