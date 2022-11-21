#pragma once

#include <complex>
#include <functional>
#include <memory>

#include "bluebild/config.h"
#include "bluebild/context.hpp"

namespace bluebild {

class BLUEBILD_EXPORT SS {

public:

    SS(const Context &ctx, const double wl,
       const size_t Nl, const size_t Nh, const size_t Nw,
       const double* pix_grid, double* stats_std_cum, double* stats_lsq_cum);

    void execute(const double* d, const std::complex<double>* v, const double* xyz,
                 const std::complex<double>* w, const std::size_t* c_idx,
                 const size_t Na, const size_t Ne, const size_t Nb, const bool d2h);
    
private:
  std::unique_ptr<void, std::function<void(void *)>> ss_;
};

class BLUEBILD_EXPORT SSf {

public:

    SSf(const Context &ctx, const float wl,
        const size_t Nl, const size_t Nh, const size_t Nw,
        const float* pix_grid, float* stats_std_cum, float* stats_lsq_sum);

    void execute(const float* d, const std::complex<float>* v, const float* xyz,
                 const std::complex<float>* w, const std::size_t* c_idx,
                 const size_t Na, const size_t Ne, const size_t Nb, const bool d2h);

private:
  std::unique_ptr<void, std::function<void(void *)>> ss_;
};

} // namespace bluebild
