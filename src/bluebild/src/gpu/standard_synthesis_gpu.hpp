#pragma once

#include "bluebild/config.h"
#include "context_internal.hpp"
#include "memory/buffer.hpp"
#include "gpu/util/gpu_runtime_api.hpp"


namespace bluebild {

template <typename T> class StandardSynthesisGPU {
public:
  StandardSynthesisGPU(std::shared_ptr<ContextInternal> ctx,
                       std::size_t nAntenna, std::size_t nBeam,
                       std::size_t nIntervals, std::size_t nFilter,
                       const BluebildFilter *filterHost, std::size_t nPixel,
                       const T *pixelX, const T *pixelY, const T *pixelZ);

  auto collect(std::size_t nEig, T wl, const T *intervalsHost,
               std::size_t ldIntervals, const gpu::ComplexType<T> *s,
               std::size_t lds, const gpu::ComplexType<T> *w, std::size_t ldw,
               T *xyz, std::size_t ldxyz) -> void;

  auto get(BluebildFilter f, T *outHostOrDevice, std::size_t ld) -> void;

  auto context() -> ContextInternal & { return *ctx_; }

private:
  std::shared_ptr<ContextInternal> ctx_;
  const std::size_t nIntervals_, nFilter_, nPixel_, nAntenna_, nBeam_;
  BufferType<BluebildFilter> filterHost_;
  BufferType<T> pixelX_, pixelY_, pixelZ_;
  BufferType<T> img_;
};

} // namespace bluebild
