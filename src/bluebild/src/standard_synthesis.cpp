#include <complex>
#include <optional>
#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "bluebild/standard_synthesis.hpp"
#include "host/ss_host.hpp"
#include "host/standard_synthesis_host.hpp"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
#include "gpu/standard_synthesis_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/util/gpu_util.hpp"
#endif

namespace bluebild {

template <typename T> struct StandardSynthesisInternal {
  StandardSynthesisInternal(const std::shared_ptr<ContextInternal> &ctx,
                            std::size_t nAntenna, std::size_t nBeam,
                            std::size_t nIntervals, std::size_t nFilter,
                            const BluebildFilter *filter, std::size_t nPixel,
                            const T *pixelX, const T *pixelY, const T *pixelZ)
      : ctx_(ctx), nAntenna_(nAntenna), nBeam_(nBeam), nIntervals_(nIntervals),
        nPixel_(nPixel) {
    if (ctx_->processing_unit() == BLUEBILD_PU_CPU) {
      planHost_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter,
                        nPixel, pixelX, pixelY, pixelZ);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<T> pixelXBuffer, pixelYBuffer, pixelZBuffer;
      auto pixelXDevice = pixelX;
      auto pixelYDevice = pixelY;
      auto pixelZDevice = pixelZ;

      if (!is_device_ptr(pixelX)) {
        pixelXBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        pixelXDevice = pixelXBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            pixelXBuffer.get(), pixelX, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(pixelY)) {
        pixelYBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        pixelYDevice = pixelYBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            pixelYBuffer.get(), pixelY, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }
      if (!is_device_ptr(pixelZ)) {
        pixelZBuffer = create_buffer<T>(ctx_->allocators().gpu(), nPixel);
        pixelZDevice = pixelZBuffer.get();
        gpu::check_status(gpu::memcpy_async(
            pixelZBuffer.get(), pixelZ, nPixel * sizeof(T),
            gpu::flag::MemcpyHostToDevice, ctx_->gpu_stream()));
      }

      planGPU_.emplace(ctx_, nAntenna, nBeam, nIntervals, nFilter, filter,
                       nPixel, pixelXDevice, pixelYDevice, pixelZDevice);
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));
#else
      throw GPUSupportError();
#endif
    }
  }

  void collect(std::size_t nEig, T wl, const T *intervals,
               std::size_t ldIntervals, const std::complex<T> *s,
               std::size_t lds, const std::complex<T> *w, std::size_t ldw,
               const T *xyz, std::size_t ldxyz) {
    if (planHost_) {
      planHost_.value().collect(nEig, wl, intervals, ldIntervals, s, lds, w,
                                ldw, xyz, ldxyz);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      BufferType<gpu::ComplexType<T>> wBuffer, sBuffer;
      BufferType<T> xyzBuffer;

      auto sDevice = reinterpret_cast<const gpu::ComplexType<T> *>(s);
      auto ldsDevice = lds;
      auto wDevice = reinterpret_cast<const gpu::ComplexType<T> *>(w);
      auto ldwDevice = ldw;

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

      // Always copy xyz, even when on device, since it will be overwritten
      xyzBuffer = create_buffer<T>(ctx_->allocators().gpu(), 3 * nAntenna_);
      auto ldxyzDevice = nAntenna_;
      auto xyzDevice = xyzBuffer.get();
      gpu::check_status(
          gpu::memcpy_2d_async(xyzBuffer.get(), nAntenna_ * sizeof(T), xyz,
                               ldxyz * sizeof(T), nAntenna_ * sizeof(T), 3,
                               gpu::flag::MemcpyDefault, ctx_->gpu_stream()));

      // sync before call, such that host memory can be safely discarded by
      // caller, while computation is continued asynchronously
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

      planGPU_->collect(nEig, wl, intervals, ldIntervals, sDevice, ldsDevice,
                        wDevice, ldwDevice, xyzDevice, ldxyzDevice);
#else
      throw GPUSupportError();
#endif
    }
  }

  auto get(BluebildFilter f, T *out, std::size_t ld) -> void {
    if (planHost_) {
      planHost_.value().get(f, out, ld);
    } else {
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
      planGPU_->get(f, out, ld);
      gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));
#else
      throw GPUSupportError();
#endif
    }
  }

  std::shared_ptr<ContextInternal> ctx_;
  std::size_t nAntenna_, nBeam_, nIntervals_, nPixel_;
  std::optional<StandardSynthesisHost<T>> planHost_;
#if defined(BLUEBILD_CUDA) || defined(BLUEBILD_ROCM)
  std::optional<StandardSynthesisGPU<T>> planGPU_;
#endif
};

template <typename T>
StandardSynthesis<T>::StandardSynthesis(
    Context &ctx, std::size_t nAntenna, std::size_t nBeam,
    std::size_t nIntervals, std::size_t nFilter, const BluebildFilter *filter,
    std::size_t nPixel, const T *pixelX, const T *pixelY, const T *pixelZ)
    : plan_(new StandardSynthesisInternal<T>(
                InternalContextAccessor::get(ctx), nAntenna, nBeam, nIntervals,
                nFilter, filter, nPixel, pixelX, pixelY, pixelZ),
            [](auto &&ptr) {
              delete reinterpret_cast<StandardSynthesisInternal<T> *>(ptr);
            }) {}

template <typename T>
auto StandardSynthesis<T>::collect(std::size_t nEig, T wl, const T *intervals,
                                   std::size_t ldIntervals,
                                   const std::complex<T> *s, std::size_t lds,
                                   const std::complex<T> *w, std::size_t ldw,
                                   const T *xyz, std::size_t ldxyz) -> void {

  reinterpret_cast<StandardSynthesisInternal<T> *>(plan_.get())
      ->collect(nEig, wl, intervals, ldIntervals, s, lds, w, ldw, xyz, ldxyz);
}

template <typename T>
auto StandardSynthesis<T>::get(BluebildFilter f, T *out, std::size_t ld)
    -> void {
  reinterpret_cast<StandardSynthesisInternal<T> *>(plan_.get())
      ->get(f, out, ld);
}

template class BLUEBILD_EXPORT StandardSynthesis<double>;

template class BLUEBILD_EXPORT StandardSynthesis<float>;

} // namespace bluebild
