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
void SSHost<double>::execute(const int iter) {
    printf("!!!!! do something here !!!!!\n");
}

template <>
SSHost<float>::SSHost(const size_t N, const float wl, const float* x) {

    ss_ = ssType();
}

template <>
void SSHost<float>::execute(const int iter) {
    printf("!!!!! do something here 2 !!!!!\n");
}

} // namespace bluebild
