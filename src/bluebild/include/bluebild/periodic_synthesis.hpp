#pragma once

#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/context.hpp"

namespace bluebild {

template <typename T>
class BLUEBILD_EXPORT PeriodicSynthesis {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

  PeriodicSynthesis(Context &ctx, T tol, int nAntenna,
                    int nBeam, int nIntervals,
                    int nFilter, const BluebildFilter *filter,
                    int nPixel, const T *lmnX, const T *lmnY,
                    const T *lmnZ);

  auto collect(int nEig, T wl, const T *intervals,
               int ldIntervals, const std::complex<T> *s,
               int lds, const std::complex<T> *w, int ldw,
               const T *xyz, int ldxyz, const T *uvwX, const T *uvwY,
               const T *uvwZ, const std::complex<T> *prephase) -> void;

  auto get(BluebildFilter f, T* out, int ld) -> void;

private:
  std::unique_ptr<void, std::function<void(void *)>> plan_;
};

} // namespace bluebild
