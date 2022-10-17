#pragma once

#include <complex>
#include <cstddef>
#include <cstring>

#include "bluebild/bluebild.hpp"
#include "bluebild/config.h"

namespace bluebild {

// Compute the location of the interval [a, b] within the descending array D.
// Returns the first index and size. Assuming n is small -> linear search should
// suffice
template<typename T>
auto find_interval_indices(std::size_t n, const T* D, T a, T b) -> std::tuple<std::size_t, std::size_t> {
  if (!n)
    return {0, 0};
  std::size_t l = n;
  std::size_t r = 0;

  for(std::size_t i =0; i < n; ++i) {
    const auto value = D[i];
    if(value <= b && value >= a) {
        if(i < l) l = i;
        if(i > r) r = i;
    }
  }

  return {l, l <= r ? r - l + 1 : 0};
}

template <typename T>
auto apply_filter(BluebildFilter f, std::size_t nEig, const T* D, T* DFiltered) -> void{
  switch(f) {
  case BLUEBILD_FILTER_STD: {
    for (std::size_t i = 0; i < nEig; ++i) {
      DFiltered[i] = 1;
    }
    break;
  }
  case BLUEBILD_FILTER_SQRT: {
    for (std::size_t i = 0; i < nEig; ++i) {
      DFiltered[i] = std::sqrt(D[i]);
    }
    break;
  }
  case BLUEBILD_FILTER_INV: {
    for (std::size_t i = 0; i < nEig; ++i) {
      const auto value = D[i];
      if (value)
        DFiltered[i] = 1 / D[i];
      else
        DFiltered[i] = 0;
    }
    break;
  }
  case BLUEBILD_FILTER_LSQ: {
    std::memcpy(DFiltered, D, nEig * sizeof(T));
    break;
  }
  }
}

}  // namespace bluebild
