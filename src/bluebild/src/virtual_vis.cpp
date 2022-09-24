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
#include "gpu/gram_matrix_gpu.hpp"
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
