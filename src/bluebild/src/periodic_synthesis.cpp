#include <complex>
#include <optional>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/periodic_synthesis.hpp"

#include "host/periodic_synthesis_host.hpp"
#include "context_internal.hpp"

#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "memory/buffer.hpp"
#include "gpu/periodic_synthesis_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T> struct PeriodicSynthesisInternal {
  PeriodicSynthesisInternal(const std::shared_ptr<ContextInternal> &ctx, T tol,
                            int nAntenna, int nBeam, int nIntervals,
                            int nFilter, const BluebildFilter *filter,
                            int nPixel, const T *lmnX, const T *lmnY,
                            const T *lmnZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals),
        nPixel_(nPixel) {
    if (ctx_->processing_unit() == BLUEBILD_PU_CPU) {
      planHost_.emplace(ctx_, tol, nAntenna, nBeam, nIntervals, nFilter, filter,
                        nPixel, lmnX, lmnY, lmnZ);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<T> lmnXBuffer, lmnYBuffer, lmnZBuffer;
      auto lmnXDevice = lmnX;
      auto lmnYDevice = lmnY;
      auto lmnZDevice = lmnZ;

      if (!is_device_ptr(lmnX)) {
        lmnXBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        lmnXDevice = lmnXBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            lmnXBuffer.get(), lmnX, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(lmnY)) {
        lmnYBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        lmnYDevice = lmnYBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            lmnYBuffer.get(), lmnY, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(lmnZ)) {
        lmnZBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        lmnZDevice = lmnZBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            lmnZBuffer.get(), lmnZ, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }

      planGPU_.emplace(ctx_, tol, nAntenna, nBeam, nIntervals, nFilter, filter,
                       nPixel, lmnXDevice, lmnYDevice, lmnZDevice);
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));
#else
      throw GPUSupportError();
#endif
    }
  }

  void collect(int nEig, T wl, const T *intervals, int ldIntervals,
               const std::complex<T> *s, int lds, const std::complex<T> *w,
               int ldw, const T *xyz, int ldxyz, const T *uvwX, const T *uvwY,
               const T *uvwZ) {
    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w,
                                ldw, xyz, ldxyz, uvwX, uvwY, uvwZ);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<gpu::ComplexType<T>> wBuffer, sBuffer;
      BufferType<T> xyzBuffer, uvwXBuffer, uvwYBuffer, uvwZBuffer;

      auto sDevice = reinterpret_cast<const gpu::ComplexType<T> *>(s);
      auto ldsDevice = lds;
      auto wDevice = reinterpret_cast<const gpu::ComplexType<T> *>(w);
      auto ldwDevice = ldw;
      auto xyzDevice = xyz;
      auto ldxyzDevice = ldxyz;
      auto uvwXDevice = uvwX;
      auto uvwYDevice = uvwY;
      auto uvwZDevice = uvwZ;

      if (s && !is_device_ptr(w)) {
        sBuffer = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                                     nBeam_ * nBeam_);
        ldsDevice = nBeam_;
        sDevice = sBuffer.get();
        gpu::check_status(gpu::memcpy_2d_async(
            sBuffer.get(), nBeam_ * sizeof(gpu::ComplexType<T>), s,
            lds * sizeof(gpu::ComplexType<T>),
            nBeam_ * sizeof(gpu::ComplexType<T>), nBeam_,
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(w)) {
        wBuffer = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                                     nAntenna_ * nBeam_);
        ldwDevice = nAntenna_;
        wDevice = wBuffer.get();
        gpu::check_status(gpu::memcpy_2d_async(
            wBuffer.get(), nAntenna_ * sizeof(gpu::ComplexType<T>), w,
            ldw * sizeof(gpu::ComplexType<T>),
            nAntenna_ * sizeof(gpu::ComplexType<T>), nBeam_,
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(xyz)) {
        xyzBuffer = create_buffer<T>(ctx_->allocators().gpu(), 3 * nAntenna_);
        ldxyzDevice = nAntenna_;
        xyzDevice = xyzBuffer.get();
        gpu::check_status(gpu::memcpy_2d_async(
            xyzBuffer.get(), nAntenna_ * sizeof(T), xyz, ldxyz * sizeof(T),
            nAntenna_ * sizeof(T), 3, gpu::flag::MemcpyHostToDevice,
            ctx_->gpu_stream()));
      }
      if (!is_device_ptr(uvwX)) {
        uvwXBuffer = create_buffer<T>(ctx_->allocators().gpu(), nAntenna_ * nAntenna_);
        uvwXDevice = uvwXBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            uvwXBuffer.get(), uvwX, nAntenna_ * nAntenna_ * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(uvwY)) {
        uvwYBuffer = create_buffer<T>(ctx_->allocators().gpu(), nAntenna_ * nAntenna_);
        uvwYDevice = uvwYBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            uvwYBuffer.get(), uvwY, nAntenna_ * nAntenna_ * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(uvwZ)) {
        uvwZBuffer = create_buffer<T>(ctx_->allocators().gpu(), nAntenna_ * nAntenna_);
        uvwZDevice = uvwZBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            uvwZBuffer.get(), uvwZ, nAntenna_ * nAntenna_ * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }

      // sync before call, such that host memory can be safely discarded by
      // caller, while computation is continued asynchronously
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

      planGPU_->collect(nEig, wl, intervals, ldIntervals, sDevice, ldsDevice,
                        wDevice, ldwDevice, xyzDevice, ldxyzDevice, uvwXDevice,
                        uvwYDevice, uvwZDevice);
#else
      throw GPUSupportError();
#endif
    }
  }

  auto get(BluebildFilter f, T *out, int ld) -> void {
    if (planHost_) {
      planHost_.value().get(f, out, ld);
    } else {
      // TODO
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      planGPU_->get(f, out, ld);
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  int nAntenna_, nBeam_, nIntervals_, nPixel_;
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
auto PeriodicSynthesis<T>::collect(int nEig, T wl, const T *intervals,
                                   int ldIntervals, const std::complex<T> *s,
                                   int lds, const std::complex<T> *w, int ldw,
                                   const T *xyz, int ldxyz, const T *uvwX,
                                   const T *uvwY, const T *uvwZ) -> void {

  reinterpret_cast<PeriodicSynthesisInternal<T> *>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz,
                uvwX, uvwY, uvwZ);
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
