#include <complex>
#include <functional>
#include <memory>
#include <cstring>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "gpu/eigensolver_gpu.hpp"
#include "gpu/gram_matrix_gpu.hpp"
#include "gpu/kernels/add_vector.hpp"
#include "gpu/nufft_3d3_gpu.hpp"
#include "gpu/nufft_synthesis_gpu.hpp"
#include "gpu/util/gpu_runtime_api.hpp"
#include "gpu/virtual_vis_gpu.hpp"

namespace bluebild {

template <typename T>
NufftSynthesisGPU<T>::NufftSynthesisGPU(
    std::shared_ptr<ContextInternal> ctx, T tol, std::size_t nAntenna,
    std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
    const BluebildFilter *filterHost, std::size_t nPixel, const T *lmnX,
    const T *lmnY, const T *lmnZ)
    : ctx_(std::move(ctx)), tol_(tol), nIntervals_(nIntervals),
      nFilter_(nFilter), nPixel_(nPixel), nAntenna_(nAntenna), nBeam_(nBeam),
      inputCount_(0) {
  filterHost_ =
      create_buffer<BluebildFilter>(ctx_->allocators().host(), nFilter_);
  std::memcpy(filterHost_.get(), filterHost, sizeof(BluebildFilter) * nFilter_);
  lmnX_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(lmnX_.get(), lmnX, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));
  lmnY_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(lmnY_.get(), lmnY, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));
  lmnZ_ = create_buffer<T>(ctx_->allocators().gpu(), nPixel_);
  gpu::check_status(gpu::memcpy_async(lmnZ_.get(), lmnZ, sizeof(T) * nPixel_,
                                      gpu::flag::MemcpyDeviceToDevice,
                                      ctx_->gpu_stream()));

  // use at most 33% of memory more accumulation, but not more than 200
  // iterations. TODO: find optimum
  std::size_t freeMem, totalMem;
  gpu::mem_get_info(&freeMem, &totalMem);
  nMaxInputCount_ = (totalMem / 3) / (nIntervals_ * nFilter_ * nAntenna_ *
                                      nAntenna_ * sizeof(std::complex<T>));
  nMaxInputCount_ =
      std::min<std::size_t>(std::max<std::size_t>(1, nMaxInputCount_), 200);

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  virtualVis_ = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                                   virtualVisBufferSize);
  uvwX_ = create_buffer<T>(ctx_->allocators().gpu(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwY_ = create_buffer<T>(ctx_->allocators().gpu(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwZ_ = create_buffer<T>(ctx_->allocators().gpu(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);

  img_ = create_buffer<T>(ctx_->allocators().gpu(),
                          nPixel_ * nIntervals_ * nFilter_);
  gpu::memset_async(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T),
                    ctx_->gpu_stream());
}

template <typename T>
auto NufftSynthesisGPU<T>::collect(
    std::size_t nEig, T wl, const T *intervals, std::size_t ldIntervals,
    const gpu::ComplexType<T> *s, std::size_t lds, const gpu::ComplexType<T> *w,
    std::size_t ldw, const T *xyz, std::size_t ldxyz, const T *uvw,
    std::size_t lduvw) -> void {

  // store coordinates
  gpu::check_status(
      gpu::memcpy_async(uvwX_.get() + inputCount_ * nAntenna_ * nAntenna_, uvw,
                        sizeof(T) * nAntenna_ * nAntenna_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));
  gpu::check_status(
      gpu::memcpy_async(uvwY_.get() + inputCount_ * nAntenna_ * nAntenna_,
                        uvw + lduvw, sizeof(T) * nAntenna_ * nAntenna_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));
  gpu::check_status(
      gpu::memcpy_async(uvwZ_.get() + inputCount_ * nAntenna_ * nAntenna_,
                        uvw + 2 * lduvw, sizeof(T) * nAntenna_ * nAntenna_,
                        gpu::flag::MemcpyDeviceToDevice, ctx_->gpu_stream()));

  auto v = create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(),
                                              nBeam_ * nEig);
  auto d = create_buffer<T>(ctx_->allocators().gpu(), nEig);

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

  auto virtVisPtr = virtualVis_.get() + inputCount_ * nAntenna_ * nAntenna_;

  virtual_vis_gpu(*ctx_, nFilter_, filterHost_.get(), nIntervals_, intervals,
                  ldIntervals, nEig, d.get(), nAntenna_, v.get(), nBeam_,
                  nBeam_, w, ldw, virtVisPtr,
                  nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
                  nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  ++inputCount_;
  if (inputCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T> auto NufftSynthesisGPU<T>::computeNufft() -> void {
  if (inputCount_) {
    auto output =
        create_buffer<gpu::ComplexType<T>>(ctx_->allocators().gpu(), nPixel_);
    auto outputPtr = output.get();
    gpu::check_status(gpu::stream_synchronize(
        ctx_->gpu_stream())); // cufinufft cannot be asigned a stream
    Nufft3d3GPU<T> transform(1, tol_, 1, nAntenna_ * nAntenna_ * inputCount_,
                             uvwX_.get(), uvwY_.get(), uvwZ_.get(), nPixel_,
                             lmnX_.get(), lmnY_.get(), lmnZ_.get());

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_;
        assert(i * ldVirtVis1 + j * ldVirtVis2 +
                   nAntenna_ * nAntenna_ * inputCount_ <=
               virtualVisBufferSize);
        transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2,
                          outputPtr);

        // use default stream to match cufiNUFFT
        gpu::add_vector_real<T>(nullptr, nPixel_, outputPtr, imgPtr);
      }
    }
  }

  gpu::check_status(
      gpu::stream_synchronize(nullptr)); // cufinufft cannot be asigned a stream
  inputCount_ = 0;
}

template <typename T>
auto NufftSynthesisGPU<T>::get(BluebildFilter f, T *outHostOrDevice,
                               std::size_t ld) -> void {
  computeNufft(); // make sure all input has been processed

  std::size_t index = nFilter_;
  const BluebildFilter *filterPtr = filterHost_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
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

template class NufftSynthesisGPU<float>;
template class NufftSynthesisGPU<double>;

} // namespace bluebild
