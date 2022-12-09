#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/exceptions.hpp"
#include "host/ss_host.hpp"

namespace bluebild {

template <>
SSHost<double>::SSHost(const size_t N, const double wl, const double* x) {
  ss_ = ssType();
}

template <>
SSHost<float>::SSHost(const size_t N, const float wl, const float* x) {
    ss_ = ssType();
}

} // namespace bluebild
