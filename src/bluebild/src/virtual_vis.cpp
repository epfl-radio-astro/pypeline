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
virtual_vis(Context &ctx, int nFilter, const BluebildFilter *filter,
            int nIntervals, const T *intervals, int ldIntervals, int nEig,
            const T *D, int nAntenna, const std::complex<T> *V, int ldv,
            int nBeam, const std::complex<T> *W, int ldw,
            std::complex<T> *virtVis, int ldVirtVis1, int ldVirtVis2,
            int ldVirtVis3) -> void {
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
    int ldvDevice = ldv;
    int ldwDevice = ldw;
    int ldVirtVis1Device = ldVirtVis1;
    int ldVirtVis2Device = ldVirtVis2;
    int ldVirtVis3Device = ldVirtVis3;

    // copy input if required
    if (!is_device_ptr(D)) {
      dBuffer =
          create_buffer<T>(ctxInternal.allocators().gpu(), nEig);
      dDevice = dBuffer.get();
      gpu::check_status(gpu::memcpy_async(dBuffer.get(), D, nEig * sizeof(T),
                                          gpu::flag::MemcpyHostToDevice,
                                          ctxInternal.gpu_stream()));
    }

    if (!is_device_ptr(V)) {
      const auto vRows = W ? nBeam : nAntenna;
      vBuffer = create_buffer<gpu::ComplexType<T>>(ctxInternal.allocators().gpu(), vRows * nEig);
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
      for (int i = 0; i < nFilter; ++i) {
        for (int j = 0; j < nIntervals; ++j) {
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
// BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_s(BluebildContext ctx, int m,
//                                                      int n, const void *w,
//                                                      int ldw, const float *xyz,
//                                                      int ldxyz, float wl,
//                                                      void *g, int ldg) {
//   if (!ctx) {
//     return BLUEBILD_INVALID_HANDLE_ERROR;
//   }
//   try {
//     gram_matrix<float>(*reinterpret_cast<Context *>(ctx), m, n,
//                        reinterpret_cast<const std::complex<float> *>(w), ldw,
//                        xyz, ldxyz, wl,
//                        reinterpret_cast<std::complex<float> *>(g), ldg);
//   } catch (const bluebild::GenericError &e) {
//     return e.error_code();
//   } catch (...) {
//     return BLUEBILD_UNKNOWN_ERROR;
//   }
//   return BLUEBILD_SUCCESS;
// }

// BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_d(BluebildContext ctx, int m,
//                                                      int n, const void *w,
//                                                      int ldw, const double *xyz,
//                                                      int ldxyz, double wl,
//                                                      void *g, int ldg) {
//   if (!ctx) {
//     return BLUEBILD_INVALID_HANDLE_ERROR;
//   }
//   try {
//     gram_matrix<double>(*reinterpret_cast<Context *>(ctx), m, n,
//                         reinterpret_cast<const std::complex<double> *>(w), ldw,
//                         xyz, ldxyz, wl,
//                         reinterpret_cast<std::complex<double> *>(g), ldg);
//   } catch (const bluebild::GenericError &e) {
//     return e.error_code();
//   } catch (...) {
//     return BLUEBILD_UNKNOWN_ERROR;
//   }
//   return BLUEBILD_SUCCESS;
// }
}



template auto
virtual_vis<float, void>(Context &ctx, int nFilter, const BluebildFilter *filter,
            int nIntervals, const float *intervals, int ldIntervals, int nEig,
            const float *D, int nAntenna, const std::complex<float> *V, int ldv,
            int nBeam, const std::complex<float> *W, int ldw,
            std::complex<float> *virtVis, int ldVirtVis1, int ldVirtVis2,
            int ldVirtVis3) -> void;
template auto
virtual_vis<double, void>(Context &ctx, int nFilter, const BluebildFilter *filter,
            int nIntervals, const double *intervals, int ldIntervals, int nEig,
            const double *D, int nAntenna, const std::complex<double> *V, int ldv,
            int nBeam, const std::complex<double> *W, int ldw,
            std::complex<double> *virtVis, int ldVirtVis1, int ldVirtVis2,
            int ldVirtVis3) -> void;

} // namespace bluebild
