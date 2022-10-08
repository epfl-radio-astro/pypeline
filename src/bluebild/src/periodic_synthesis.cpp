#include <complex>
#include <optional>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/periodic_synthesis.hpp"

#include "host/periodic_synthesis_host.hpp"
#include "context_internal.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "gpu/periodic_syhtesis_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T> struct PeriodicSynthesisInternal {
  PeriodicSynthesisInternal(const std::shared_ptr<ContextInternal> &ctx, T tol,
                            int nAntenna, int nBeam,
                            int nIntervals, int nFilter,
                            const BluebildFilter *filter, int nPixel,
                            const T *lmnX, const T *lmnY, const T *lmnZ) {
    if (ctx->processing_unit() == BLUEBILD_PU_CPU) {
      planHost_.emplace(ctx, tol, nAntenna, nBeam, nIntervals, nFilter, filter,
                        nPixel, lmnX, lmnY, lmnZ);
    } else {
      // TODO
    }
  }

  void collect(int nEig, T wl, const T *intervals,
               int ldIntervals, const std::complex<T> *s,
               int lds, const std::complex<T> *w, int ldw,
               const T *xyz, int ldxyz, const T *uvwX, const T *uvwY,
               const T *uvwZ, const std::complex<T> *prephase) {
    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w,
                                ldw, xyz, ldxyz, uvwX, uvwY, uvwZ, prephase);
    } else {
      // TODO
    }
  }

  auto get(BluebildFilter f, T *out, int ld) -> void {
    if (planHost_) {
      planHost_.value().get(f, out, ld);
    } else {
      // TODO
    }
  }

  std::optional<PeriodicSynthesisHost<T>> planHost_;
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
  std::optional<PeriodicSynthesisGPU<T>> planGPU_;
#endif
};

template <typename T>
PeriodicSynthesis<T>::PeriodicSynthesis(
    Context &ctx, T tol, int nAntenna, int nBeam,
    int nIntervals, int nFilter, const BluebildFilter *filter,
    int nPixel, const T *lmnX, const T *lmnY, const T *lmnZ)
    : plan_(new PeriodicSynthesisInternal<T>(
                InternalContextAccessor::get(ctx), tol, nAntenna, nBeam,
                nIntervals, nFilter, filter, nPixel, lmnX, lmnY, lmnZ),
            [](auto &&ptr) {
              delete reinterpret_cast<PeriodicSynthesisInternal<T> *>(ptr);
            }) {}

template <typename T>
auto PeriodicSynthesis<T>::collect(
    int nEig, T wl, const T *intervals, int ldIntervals,
    const std::complex<T> *s, int lds, const std::complex<T> *w,
    int ldw, const T *xyz, int ldxyz, const T *uvwX,
    const T *uvwY, const T *uvwZ, const std::complex<T> *prephase) -> void {

  reinterpret_cast<PeriodicSynthesisInternal<T> *>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz,
                uvwX, uvwY, uvwZ, prephase);
}

template <typename T>
auto PeriodicSynthesis<T>::get(BluebildFilter f, T *out, int ld)
    -> void {
  reinterpret_cast<PeriodicSynthesisInternal<T> *>(plan_.get())
      ->get(f, out, ld);
}

template class BLUEBILD_EXPORT PeriodicSynthesis<double>;

template class BLUEBILD_EXPORT PeriodicSynthesis<float>;

} // namespace bluebild
