#pragma once

#include <complex>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "context_internal.hpp"
#include "memory/buffer.hpp"

namespace bluebild {

template <typename T> class PeriodicSynthesisHost {
public:
  PeriodicSynthesisHost(std::shared_ptr<ContextInternal> ctx, T tol,
                        std::size_t nAntenna, std::size_t nBeam,
                        std::size_t nIntervals, std::size_t nFilter,
                        const BluebildFilter *filter, std::size_t nPixel,
                        const T *lmnX, const T *lmnY, const T *lmnZ);

  auto collect(std::size_t nEig, T wl, const T *intervals,
               std::size_t ldIntervals, const std::complex<T> *s,
               std::size_t lds, const std::complex<T> *w, std::size_t ldw,
               const T *xyz, std::size_t ldxyz, const T *uvwX, const T *uvwY,
               const T *uvwZ) -> void;

  auto get(BluebildFilter f, T* out, std::size_t ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const T tol_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filter_;
  BufferType<T> lmnX_, lmnY_, lmnZ_;

  std::size_t nMaxInputCount_, inputCount_;
  BufferType<std::complex<T>> virtualVis_;
  BufferType<T> uvwX_, uvwY_, uvwZ_;
  BufferType<T> img_;

};

} // namespace bluebild
