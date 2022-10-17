#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"
#include "host/virtual_vis_host.hpp"


#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "eigensolver.hpp"
#include "gpu/virtual_vis_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T, typename>
BLUEBILD_EXPORT auto
virtual_vis(Context &ctx, std::size_t nFilter, const BluebildFilter *filter,
            std::size_t nIntervals, const T *intervals, std::size_t ldIntervals,
            std::size_t nEig, const T *D, std::size_t nAntenna,
            const std::complex<T> *V, std::size_t ldv, std::size_t nBeam,
            const std::complex<T> *W, std::size_t ldw, std::complex<T> *virtVis,
            std::size_t ldVirtVis1, std::size_t ldVirtVis2,
            std::size_t ldVirtVis3) -> void {
  auto &ctxInternal = *InternalContextAccessor::get(ctx);
  if (ctxInternal.processing_unit() == BLUEBILD_PU_GPU) {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
    // Syncronize with default stream. TODO: replace with event
    gpu::stream_synchronize(nullptr);

    BufferType<gpu::ComplexType<T>> vBuffer, wBuffer, virtVisBuffer;
    BufferType<T> dBuffer;
    auto dDevice = reinterpret_cast<const T *>(D);
    auto vDevice = reinterpret_cast<const gpu::ComplexType<T> *>(V);
    auto wDevice = reinterpret_cast<const gpu::ComplexType<T> *>(W);
    auto virtVisDevice = reinterpret_cast<gpu::ComplexType<T> *>(virtVis);
    std::size_t ldvDevice = ldv;
    std::size_t ldwDevice = ldw;
    std::size_t ldVirtVis1Device = ldVirtVis1;
    std::size_t ldVirtVis2Device = ldVirtVis2;
    std::size_t ldVirtVis3Device = ldVirtVis3;

    // copy input if required
    if (!is_device_ptr(D)) {
      dBuffer = create_buffer<T>(ctxInternal.allocators().gpu(), nEig);
      dDevice = dBuffer.get();
      gpu::check_status(gpu::memcpy_async(dBuffer.get(), D, nEig * sizeof(T),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctxInternal.gpu_stream()));
    }

    if (!is_device_ptr(V)) {
      const auto vRows = W ? nBeam : nAntenna;
      vBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(), vRows * nEig);
      ldvDevice = vRows;
      vDevice = vBuffer.get();
      gpu::check_status(gpu::memcpy_2d_async(
          vBuffer.get(), vRows * sizeof(gpu::ComplexType<T>), V,
          ldv * sizeof(gpu::ComplexType<T>),
          vRows * sizeof(gpu::ComplexType<T>), nEig,
          gpu::flag::MemcpyHostToDevice, ctxInternal.gpu_stream()));
    }

    if (W && !is_device_ptr(W)) {
      wBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(), nAntenna * nBeam);
      ldwDevice = nAntenna;
      wDevice = wBuffer.get();
      gpu::check_status(gpu::memcpy_2d_async(
          wBuffer.get(), nAntenna * sizeof(gpu::ComplexType<T>), W,
          ldw * sizeof(gpu::ComplexType<T>),
          nAntenna * sizeof(gpu::ComplexType<T>), nBeam,
          gpu::flag::MemcpyHostToDevice, ctxInternal.gpu_stream()));
    }

    if (!is_device_ptr(virtVis)) {
      virtVisBuffer = create_buffer<gpu::ComplexType<T>>(
          ctxInternal.allocators().gpu(),
          nFilter * nIntervals * nAntenna * nAntenna);
      ldVirtVis3 = nAntenna;
      ldVirtVis2 = nAntenna * nAntenna;
      ldVirtVis1 = nAntenna * nAntenna * nIntervals;
      virtVisDevice = virtVisBuffer.get();
    }

    virtual_vis_gpu<T>(ctxInternal, nFilter, filter, nIntervals, intervals,
                       ldIntervals, nEig, dDevice, nAntenna, vDevice, ldvDevice,
                       nBeam, wDevice, ldwDevice, virtVisDevice,
                       ldVirtVis1Device, ldVirtVis2Device, ldVirtVis3Device);

    // copy back results if required
    if (virtVisBuffer) {
      for (std::size_t i = 0; i < nFilter; ++i) {
        for (std::size_t j = 0; j < nIntervals; ++j) {
          gpu::check_status(gpu::memcpy_2d_async(
              virtVis + i * ldVirtVis1 + j * ldVirtVis2,
              ldVirtVis3 * sizeof(gpu::ComplexType<T>),
              virtVisDevice + i * ldVirtVis1Device + j * ldVirtVis2Device,
              ldVirtVis3Device * sizeof(gpu::ComplexType<T>),
              nAntenna * sizeof(gpu::ComplexType<T>), nAntenna,
              gpu::flag::MemcpyDeviceToHost, ctxInternal.gpu_stream()));
        }
      }
    }

    // syncronize with stream to be synchronous with host
    gpu::check_status(gpu::stream_synchronize(ctxInternal.gpu_stream()));
#else
    throw GPUSupportError();
#endif
  } else {
    virtual_vis_host<T>(ctxInternal, nFilter, filter, nIntervals, intervals,
                        ldIntervals, nEig, D, nAntenna, V, ldv, nBeam, W, ldw,
                        virtVis, ldVirtVis1, ldVirtVis2, ldVirtVis3);
  }
}

extern "C" {
BLUEBILD_EXPORT BluebildError bluebild_virtual_vis_s(
    BluebildContext ctx, size_t nFilter, const BluebildFilter *filter,
    size_t nIntervals, const float *intervals, size_t ldIntervals, size_t nEig,
    const float *D, size_t nAntenna, const void *V, size_t ldv, size_t nBeam,
    const void *W, size_t ldw, void *virtVis, size_t ldVirtVis1,
    size_t ldVirtVis2, size_t ldVirtVis3) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    virtual_vis<float>(*reinterpret_cast<Context *>(ctx), nFilter, filter,
                       nIntervals, intervals, ldIntervals, nEig, D, nAntenna,
                       reinterpret_cast<const std::complex<float> *>(V), ldv,
                       nBeam, reinterpret_cast<const std::complex<float> *>(W),
                       ldw, reinterpret_cast<std::complex<float> *>(virtVis),
                       ldVirtVis1, ldVirtVis2, ldVirtVis3);
  } catch (const bluebild::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}

BLUEBILD_EXPORT BluebildError bluebild_virtual_vis_d(
    BluebildContext ctx, size_t nFilter, const BluebildFilter *filter,
    size_t nIntervals, const double *intervals, size_t ldIntervals, size_t nEig,
    const double *D, size_t nAntenna, const void *V, size_t ldv, size_t nBeam,
    const void *W, size_t ldw, void *virtVis, size_t ldVirtVis1,
    size_t ldVirtVis2, size_t ldVirtVis3) {
  if (!ctx) {
    return BLUEBILD_INVALID_HANDLE_ERROR;
  }
  try {
    virtual_vis<double>(*reinterpret_cast<Context *>(ctx), nFilter, filter,
                        nIntervals, intervals, ldIntervals, nEig, D, nAntenna,
                        reinterpret_cast<const std::complex<double> *>(V), ldv,
                        nBeam,
                        reinterpret_cast<const std::complex<double> *>(W), ldw,
                        reinterpret_cast<std::complex<double> *>(virtVis),
                        ldVirtVis1, ldVirtVis2, ldVirtVis3);
  } catch (const bluebild::GenericError &e) {
    return e.error_code();
  } catch (...) {
    return BLUEBILD_UNKNOWN_ERROR;
  }
  return BLUEBILD_SUCCESS;
}
}

template auto virtual_vis<float, void>(
    Context &ctx, std::size_t nFilter, const BluebildFilter *filter,
    std::size_t nIntervals, const float *intervals, std::size_t ldIntervals,
    std::size_t nEig, const float *D, std::size_t nAntenna,
    const std::complex<float> *V, std::size_t ldv, std::size_t nBeam,
    const std::complex<float> *W, std::size_t ldw, std::complex<float> *virtVis,
    std::size_t ldVirtVis1, std::size_t ldVirtVis2, std::size_t ldVirtVis3)
    -> void;
template auto virtual_vis<double, void>(
    Context &ctx, std::size_t nFilter, const BluebildFilter *filter,
    std::size_t nIntervals, const double *intervals, std::size_t ldIntervals,
    std::size_t nEig, const double *D, std::size_t nAntenna,
    const std::complex<double> *V, std::size_t ldv, std::size_t nBeam,
    const std::complex<double> *W, std::size_t ldw,
    std::complex<double> *virtVis, std::size_t ldVirtVis1,
    std::size_t ldVirtVis2, std::size_t ldVirtVis3) -> void;

} // namespace bluebild
