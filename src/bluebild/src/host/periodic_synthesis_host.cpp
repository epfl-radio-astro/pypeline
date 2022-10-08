#include <complex>
#include <functional>
#include <memory>
#include <cstring>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "host/intensity_field_data_host.hpp"
#include "host/sensitivity_field_data_host.hpp"
#include "host/nufft_3d3_host.hpp"
#include "host/periodic_synthesis_host.hpp"
#include "host/virtual_vis_host.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T>
PeriodicSynthesisHost<T>::PeriodicSynthesisHost(
    std::shared_ptr<ContextInternal> ctx, T tol, std::size_t nAntenna,
    std::size_t nBeam, std::size_t nIntervals, std::size_t nFilter,
    const BluebildFilter *filter, std::size_t nPixel, const T *lmnX,
    const T *lmnY, const T *lmnZ)
    : ctx_(std::move(ctx)), tol_(tol), nIntervals_(nIntervals),
      nFilter_(nFilter), nPixel_(nPixel), nAntenna_(nAntenna), nBeam_(nBeam),
      inputCount_(0) {
  filter_ = create_buffer<BluebildFilter>(ctx_->allocators().host(), nFilter_);
  std::memcpy(filter_.get(), filter, sizeof(BluebildFilter) * nFilter_);
  lmnX_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(lmnX_.get(), lmnX, sizeof(T) * nPixel_);
  lmnY_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(lmnY_.get(), lmnY, sizeof(T) * nPixel_);
  lmnZ_ = create_buffer<T>(ctx_->allocators().host(), nPixel_);
  std::memcpy(lmnZ_.get(), lmnZ, sizeof(T) * nPixel_);

  nMaxInputCount_ = 50; // TODO: compute as fraction of system memory

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  virtualVis_ = create_buffer<std::complex<T>>(ctx_->allocators().host(),
                                               virtualVisBufferSize);
  std::memset(virtualVis_.get(), 0,
              virtualVisBufferSize * sizeof(std::complex<T>));
  uvwX_ = create_buffer<T>(ctx_->allocators().host(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwY_ = create_buffer<T>(ctx_->allocators().host(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);
  uvwZ_ = create_buffer<T>(ctx_->allocators().host(),
                           nAntenna_ * nAntenna_ * nMaxInputCount_);

  img_ = create_buffer<T>(ctx_->allocators().host(),
                          nPixel_ * nIntervals_ * nFilter_);
  std::memset(img_.get(), 0, nPixel_ * nIntervals_ * nFilter_ * sizeof(T));
}

template <typename T>
auto PeriodicSynthesisHost<T>::collect(
    std::size_t nEig, T wl, const T *intervals, std::size_t ldIntervals,
    const std::complex<T> *s, std::size_t lds, const std::complex<T> *w,
    std::size_t ldw, const T *xyz, std::size_t ldxyz, const T *uvwX,
    const T *uvwY, const T *uvwZ, const std::complex<T> *prephase) -> void {

  // store coordinates
  std::memcpy(uvwX_.get() + inputCount_ * nAntenna_ * nAntenna_, uvwX,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwY_.get() + inputCount_ * nAntenna_ * nAntenna_, uvwY,
              sizeof(T) * nAntenna_ * nAntenna_);
  std::memcpy(uvwZ_.get() + inputCount_ * nAntenna_ * nAntenna_, uvwZ,
              sizeof(T) * nAntenna_ * nAntenna_);

  auto v =
      create_buffer<std::complex<T>>(ctx_->allocators().host(), nBeam_ * nEig);
  auto d = create_buffer<T>(ctx_->allocators().host(), nEig);
  auto indices = create_buffer<int>(ctx_->allocators().host(), nEig);
  auto cluster =
      create_buffer<T>(ctx_->allocators().host(),
                       nIntervals_); // dummy input until
                                     // intensity_field_data_host can be updated

  if (s)
    intensity_field_data_host(*ctx_, wl, nAntenna_, nBeam_, nEig, s, lds, w,
                              ldw, xyz, ldxyz, d.get(), v.get(), nBeam_,
                              nIntervals_, cluster.get(), indices.get());
  else
    sensitivity_field_data_host(*ctx_, wl, nAntenna_, nBeam_, nEig, w, ldw, xyz,
                                ldxyz, d.get(), v.get(), nBeam_);

  auto virtVisPtr = virtualVis_.get() + inputCount_ * nAntenna_ * nAntenna_;

  const auto ldVirtVis3 = nAntenna_;
  const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
  const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

  virtual_vis_host(*ctx_, nFilter_, filter_.get(), nIntervals_, intervals,
                   ldIntervals, nEig, d.get(), nAntenna_, v.get(), nBeam_,
                   nBeam_, w, ldw, virtVisPtr,
                   nMaxInputCount_ * nIntervals_ * nAntenna_ * nAntenna_,
                   nMaxInputCount_ * nAntenna_ * nAntenna_, nAntenna_);

  const auto virtualVisBufferSize =
      nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
  for (std::size_t i = 0; i < nFilter_; ++i) {
    for (std::size_t j = 0; j < nIntervals_; ++j) {
      auto virtVisInnerPtr = virtVisPtr + i * ldVirtVis1 + j * ldVirtVis2;
      for (std::size_t k = 0; k < nAntenna_; ++k) {
        for (std::size_t l = 0; l < nAntenna_; ++l) {
          if (l != k)
            virtVisInnerPtr[k * nAntenna_ + l] =
                std::conj(virtVisInnerPtr[k * nAntenna_ + l]);
        }
      }
    }
  }
  for (std::size_t i = 0; i < nFilter_; ++i) {
    for (std::size_t j = 0; j < nIntervals_; ++j) {
      auto virtVisInnerPtr = virtVisPtr + i * ldVirtVis1 + j * ldVirtVis2;
      for (std::size_t k = 0; k < nAntenna_ * nAntenna_; ++k) {
        assert(virtVisInnerPtr - virtualVis_.get() + k < virtualVisBufferSize);
        virtVisInnerPtr[k] *= prephase[k];
      }
    }
  }

  ++inputCount_;
  if (inputCount_ >= nMaxInputCount_) {
    computeNufft();
  }
}

template <typename T> auto PeriodicSynthesisHost<T>::computeNufft() -> void {
  if (inputCount_) {
    auto output = create_buffer<std::complex<T>>(
        ctx_->allocators().host(), nPixel_ * nFilter_ * nIntervals_);
    Nufft3d3Host<T> transform(1, tol_, 1, nAntenna_ * nAntenna_ * inputCount_,
                           uvwX_.get(), uvwY_.get(), uvwZ_.get(), nPixel_,
                           lmnX_.get(), lmnY_.get(), lmnZ_.get());

    const auto ldVirtVis3 = nAntenna_;
    const auto ldVirtVis2 = nMaxInputCount_ * nAntenna_ * ldVirtVis3;
    const auto ldVirtVis1 = nIntervals_ * ldVirtVis2;

    const auto virtualVisBufferSize =
        nIntervals_ * nFilter_ * nAntenna_ * nAntenna_ * nMaxInputCount_;
    const auto imgSize = nPixel_ * nIntervals_ * nFilter_;


    for (std::size_t i = 0; i < nFilter_; ++i) {
      for (std::size_t j = 0; j < nIntervals_; ++j) {
        assert(i * ldVirtVis1 + j * ldVirtVis2 + nAntenna_ * nAntenna_ * inputCount_ <= virtualVisBufferSize);
        transform.execute(virtualVis_.get() + i * ldVirtVis1 + j * ldVirtVis2,
                          output.get() + i * (nPixel_ * nIntervals_) + j * nPixel_);

        auto imgPtr = img_.get() + (j + i * nIntervals_) * nPixel_;
        auto outputPtr = output.get();
        for (std::size_t k = 0; k < nPixel_; ++k) {
          assert(imgPtr - img_.get() + k < imgSize);
          imgPtr[k] += outputPtr[k].real();
        }
      }
    }
  }

  inputCount_ = 0;
}

template <typename T>
auto PeriodicSynthesisHost<T>::get(BluebildFilter f, T *out, std::size_t ld)
    -> void {
  computeNufft(); // make sure all input has been processed

  std::size_t index = nFilter_;
  const BluebildFilter *filterPtr = filter_.get();
  for (std::size_t i = 0; i < nFilter_; ++i) {
    if (filterPtr[i] == f) {
      index = i;
      break;
    }
  }

  for (std::size_t i = 0; i < nIntervals_; ++i) {
    std::memcpy(out + i * ld,
                img_.get() + index * nIntervals_ * nPixel_ + i * nPixel_,
                sizeof(T) * nPixel_);
  }
}

template class PeriodicSynthesisHost<float>;
template class PeriodicSynthesisHost<double>;

} // namespace bluebild
