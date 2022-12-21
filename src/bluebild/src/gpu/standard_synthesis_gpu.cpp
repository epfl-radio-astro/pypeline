#include <complex>
#include <functional>
#include <memory>
#include <cstring>
#include <cstddef>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "gpu/gram_matrix_gpu.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/kernels/apply_filter.hpp"
#include "gpu/kernels/center_vector.hpp"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/kernels/scale_matrix.hpp"
#include "gpu/standard_synthesis_gpu.hpp"
#include "gpu/util/gpu_blas_api.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "util.hpp"

namespace bluebild {

template <typename T>
StandardSynthesisGPU<T>::StandardSynthesisGPU(
    std::shared_ptr<ContextInternal> ctx, std::size_t nAntenna,
    std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
    const BluebildFilter *filterHost, std::size_t nPixel, const T *pixelX,
    const T *pixelY, const T *pixelZ)
    : ctx_(std::move(ctx)), nIntervals_(nIntervals), nFilter_(nFilter),
      nPixel_(nPixel), nAntenna_(nAntenna), nBeam_(nBeam) {
  filterHost_ =
      create_buffer<BluebildFilter>(ctx_->allocators().host(), nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BluebildFilter) * nFilter_);
  pixelX_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(
      gpu::memcpy_async(pixelX_.get(), pixelX, sizeof(T) * nPixel_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));
  pixelY_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(
      gpu::memcpy_async(pixelY_.get(), pixelY, sizeof(T) * nPixel_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));
  pixelZ_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(
      gpu::memcpy_async(pixelZ_.get(), pixelZ, sizeof(T) * nPixel_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));

  img_ = create_buffer<T>(ctx_->allocators().gpu(),
                          nPixel_ * nIntervals_ * nFilter_);
  gpu::memset_async(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T),
                    ctx_->gpu_stream());
}

template <typename T>
auto StandardSynthesisGPU<T>::collect(
    std::size_t nEig, T wl, const T *intervalsHost, std::size_t ldIntervals,
    const gpu::ComplexType<T> *s, std::size_t lds, const gpu::ComplexType<T> *w,
    std::size_t ldw, T *xyz, std::size_t ldxyz) -> void {

  auto v = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                              nBeam_ * nEig);
  auto d = create_buffer<T>(ctx_->allocators().gpu(), nEig);
  auto vUnbeam = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                                    nAntenna_ * nEig);
  auto unlayeredStats =
      create_buffer<T>(ctx_->allocators().gpu(), nPixel_ * nEig);

  // Center coordinates for much better performance of cos / sin
  {
    std::size_t worksize =
        gpu::center_vector_get_worksize<T>(ctx_->gpu_stream(), nAntenna_);
    auto workBuffer = create_buffer<char>(ctx_->allocators().gpu(), worksize);
    for (std::size_t i = 0; i < 3; ++i) {
      gpu::center_vector<T>(ctx_->gpu_stream(), nAntenna_, xyz + i * ldxyz,
                            worksize, workBuffer.get());
    }
  }

  {
    auto g = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                                nBeam_ * nBeam_);

    gram_matrix_gpu<T>(*ctx_, nAntenna_, nBeam_, w, ldw, xyz, ldxyz, wl,
                       g.get(), nBeam_);

    std::size_t nEigOut = 0;
    // Note different order of s and g input
    if (s)
      eigh_gpu<T>(*ctx_, nBeam_, nEig, s, lds, g.get(), nBeam_, &nEigOut,
                  d.get(), v.get(), nBeam_);
    else
      eigh_gpu<T>(*ctx_, nBeam_, nEig, g.get(), nBeam_, nullptr, 0, &nEigOut,
                  d.get(), v.get(), nBeam_);
  }

  auto DBufferHost = create_buffer<T>(ctx_->allocators().pinned(), nEig);
  auto DFilteredBufferHost = create_buffer<T>(ctx_->allocators().host(), nEig);
  gpu::check_status(
      gpu::memcpy_async(DBufferHost.get(), d.get(), nEig * sizeof(T),
                        gpu::flag::MemcpyDeviceToHost, ctx_->gpu_stream()));
  // Make sure D is available on host
  gpu::check_status(gpu::stream_synchronize(ctx_->gpu_stream()));

  gpu::ComplexType<T> one{1, 0};
  gpu::ComplexType<T> zero{0, 0};
  gpu::blas::check_status(gpu::blas::gemm(
      ctx_->gpu_blas_handle(), gpu::blas::operation::None,
      gpu::blas::operation::None, nAntenna_, nEig, nBeam_, &one, w, ldw,
      v.get(), nBeam_, &zero, vUnbeam.get(), nAntenna_));

  T alpha = 2.0 * M_PI / wl;
  gemmexp_gpu<T>(ctx_->gpu_stream(), nEig, nPixel_, nAntenna_, alpha,
                 vUnbeam.get(), nAntenna_, xyz, ldxyz, pixelX_.get(),
                 pixelY_.get(), pixelZ_.get(), unlayeredStats.get(), nPixel_);

  // cluster eigenvalues / vectors based on invervals
  for (std::size_t idxFilter = 0;
       idxFilter < static_cast<std::size_t>(nFilter_); ++idxFilter) {
    apply_filter(filterHost_.get()[idxFilter], nEig, DBufferHost.get(),
                 DFilteredBufferHost.get());

    for (std::size_t idxInt = 0; idxInt < static_cast<std::size_t>(nIntervals_);
         ++idxInt) {
      std::size_t start, size;
      std::tie(start, size) = find_interval_indices(
          nEig, DBufferHost.get(),
          intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals)],
          intervalsHost[idxInt * static_cast<std::size_t>(ldIntervals) + 1]);

      auto imgCurrent =
          img_.get() + (idxFilter * nIntervals_ + idxInt) * nPixel_;
      for (std::size_t idxEig = start; idxEig < start + size; ++idxEig) {
        const auto scale = DFilteredBufferHost.get()[idxEig];
        auto unlayeredStatsCurrent = unlayeredStats.get() + nPixel_ * idxEig;
        gpu::blas::check_status(
            gpu::blas::axpy(ctx_->gpu_blas_handle(), nPixel_, &scale,
                            unlayeredStatsCurrent, 1, imgCurrent, 1));
      }
    }
  }
}

template <typename T>
auto StandardSynthesisGPU<T>::get(BluebildFilter f, T *outHostOrDevice,
                                  std::size_t ld) -> void {
  std::size_t index = nFilter_;
  const BluebildFilter *filterPtr = filterHost_.get();
  for (std::size_t idxFilter = 0; idxFilter < nFilter_; ++idxFilter) {
    if (filterPtr[idxFilter] == f) {
      index = idxFilter;
      break;
    }
  }
  if (index == nFilter_)
    throw InvalidParameterError();

  gpu::check_status(gpu::memcpy_2d_async(
      outHostOrDevice, ld * sizeof(T),
      img_.get() + index * nIntervals_ * nPixel_, nPixel_ * sizeof(T),
      nPixel_ * sizeof(T), nIntervals_, gpu::flag::MemcpyDefault,
      ctx_->gpu_stream()));
}

template class StandardSynthesisGPU<float>;
template class StandardSynthesisGPU<double>;

} // namespace bluebild
