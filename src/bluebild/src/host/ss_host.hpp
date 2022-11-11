#pragma once

#include <complex>
#include <memory>
#include <functional>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"


namespace bluebild {

template <typename T> class SSHost {
public:
  using ssType = std::unique_ptr<void, std::function<void(void *)>>;

  SSHost(const size_t N, const T wl, const T *x);

private:
  ssType ss_;
};

} // namespace bluebild
