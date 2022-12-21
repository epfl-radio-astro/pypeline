#pragma once

#include <complex>
#include <cstddef>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "memory/buffer.hpp"
#include "context_internal.hpp"

namespace bluebild {

template <typename T> class StandardSynthesisHost {
public:
  StandardSynthesisHost(std::shared_ptr<ContextInternal> ctx,
                        std::size_t nAntenna, std::size_t nBeam,
                        std::size_t nIntervals, std::size_t nFilter,
                        const BluebildFilter *filter, std::size_t nPixel,
                        const T *pixelX, const T *pixelY, const T *pixelZ);

  auto collect(std::size_t nEig, T wl, const T *intervals,
               std::size_t ldIntervals, const std::complex<T> *s,
               std::size_t lds, const std::complex<T> *w, std::size_t ldw,
               const T *xyz, std::size_t ldxyz) -> void;

  auto get(BluebildFilter f, T* out, std::size_t ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  auto computeNufft() -> void;

  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filter_;
  BufferType<T> pixelX_, pixelY_, pixelZ_;
  BufferType<T> img_;

};

}  // namespace bluebild
