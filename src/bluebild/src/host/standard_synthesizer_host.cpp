#include "host/standard_synthesizer_host.hpp"

#include <complex>
#include <complex.h>
#include <iostream>
#include <omp.h>

#include "bluebild/bluebild.h"
#include "bluebild/config.h"
#include "context_internal.hpp"
#include "host/blas_api.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer.hpp"
#include "host/gemmexp.hpp"


namespace bluebild {

template <typename T, typename U>
static
void cluster_layers(const T*  __restrict__ unlayered_stats, const T* __restrict__ d,
                    const size_t Nw, const size_t Nh, const size_t Ne, const size_t Nl,
                    const U* __restrict__ c_idx,
                    T* __restrict__ stats_std_cum, T* __restrict__ stats_lsq_cum) {

#pragma omp parallel for
    for (size_t i=0; i<Nw; i++) {
        for (size_t j=0; j<Nh; j++) {
            size_t idx_unlay = i * Nh * Ne + j * Ne;
            size_t idx_stats = i * Nh * Nl + j * Nl;
            for (size_t k=0; k<Ne; k++) {
                stats_std_cum[idx_stats + c_idx[k]] += unlayered_stats[idx_unlay + k];
                stats_lsq_cum[idx_stats + c_idx[k]] += unlayered_stats[idx_unlay + k] * d[k];
            }
        }
    }
}

template <typename T>
auto mean_center(T* __restrict__ xyz, const T* __restrict__ xyz_, const size_t N) -> void {
    auto sum_x = 0.0;
    auto sum_y = 0.0;
    auto sum_z = 0.0;
    for (size_t i=0; i<N; i++) {
        sum_x += xyz_[i];
        sum_y += xyz_[i+N];
        sum_z += xyz_[i+2*N];
    }
    auto mean_x = sum_x / static_cast<T>(N);
    auto mean_y = sum_y / static_cast<T>(N);
    auto mean_z = sum_z / static_cast<T>(N);

    for (size_t i=0; i<N; i++) {
        xyz[i]     = xyz_[i]     - mean_x;
        xyz[i+N]   = xyz_[i+N]   - mean_y;
        xyz[i+2*N] = xyz_[i+2*N] - mean_z;
    }
}

template <typename T>
auto standard_synthesizer_host(ContextInternal& ctx,
                               const T* __restrict__ d,
                               const std::complex<T>* __restrict__ v,
                               const T* __restrict__ xyz_,
                               const std::complex<T>* __restrict__ w,
                               const std::size_t* __restrict__ c_idx,
                               const std::size_t Nl,
                               const T* __restrict__ grid,
                               const T  wl,
                               const std::size_t Na,
                               const std::size_t Nb,
                               const std::size_t Nc,
                               const std::size_t Ne,
                               const std::size_t Nh,
                               const std::size_t Nw,
                               T* __restrict__ stats_std_cum,
                               T* __restrict__ stats_lsq_cum) -> void {

    T alpha = 2.0 * M_PI  / wl;

    auto xyz_buffer  = create_buffer<T>(ctx.allocators().host(), Na * 3);
    auto xyz = xyz_buffer.get();
    auto p_buffer  = create_buffer<std::complex<T>>(ctx.allocators().host(), Na * Nh * Nw);
    auto p  = p_buffer.get();
    auto pw_buffer = create_buffer<std::complex<T>>(ctx.allocators().host(), Nb * Nh * Nw);
    auto pw = pw_buffer.get();
    auto e_buffer  = create_buffer<std::complex<T>>(ctx.allocators().host(), Ne * Nh * Nw);
    auto e  = e_buffer.get();
    auto unlayered_stats_buffer = create_buffer<T>(ctx.allocators().host(), Ne * Nh * Nw);
    auto unlayered_stats = unlayered_stats_buffer.get();

    mean_center(xyz, xyz_, Na);
    
#pragma omp parallel for
    for (std::size_t i = 0; i<Nw; i++) {

        size_t idx_g  = i * Nh * Nc;
        size_t idx_p  = i * Nh * Na;
        size_t idx_pw = i * Nh * Nb;
        size_t idx_e  = i * Nh * Ne;

        gemmexp(Na, Nh, Nc, alpha, &xyz[0], Na, &grid[idx_g], Nc, &p[idx_p], Na);

        blas::gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Nb, Nh, Na, {1, 0},
                    w, Na,
                    &p[idx_p], Na,
                    {0, 0},
                    &pw[idx_pw], Nb);

        blas::gemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Ne, Nh, Nb, {1, 0},
                    v, Nb,
                    &pw[idx_pw], Nb,
                    {0, 0},
                    &e[idx_e], Ne);
    }

#pragma omp parallel for
    for (size_t i=0; i<Ne*Nh*Nw; i++) {
        unlayered_stats[i] = std::norm(e[i]);
    }

    cluster_layers(unlayered_stats, d, Nw, Nh, Ne, Nl, c_idx, stats_std_cum, stats_lsq_cum);
}

template auto standard_synthesizer_host<float>(ContextInternal& ctx,
                                               const float* __restrict__ d,
                                               const std::complex<float>* v,
                                               const float* __restrict__ xyz,
                                               const std::complex<float>* __restrict__ w,
                                               const std::size_t* __restrict__ c_idx,
                                               const std::size_t Nl,
                                               const float* __restrict__ grid,
                                               const float wl,
                                               const std::size_t Na,
                                               const std::size_t Nb,
                                               const std::size_t Nc,
                                               const std::size_t Ne,
                                               const std::size_t Nh,
                                               const std::size_t Nw,
                                               float* __restrict__ stats_std_cum,
                                               float* __restrict__ stats_lsq_cum) -> void;

template auto standard_synthesizer_host<double>(ContextInternal& ctx,
                                                const double* __restrict__ d,
                                                const std::complex<double>* __restrict__ v,
                                                const double* __restrict__ xyz,
                                                const std::complex<double>* __restrict__ w,
                                                const std::size_t* __restrict__ c_idx,
                                                const std::size_t Nl,
                                                const double* __restrict__ grid,
                                                const double wl,
                                                const std::size_t Na,
                                                const std::size_t Nb,
                                                const std::size_t Nc,
                                                const std::size_t Ne,
                                                const std::size_t Nh,
                                                const std::size_t Nw,
                                                double* __restrict__ stats_std_cum,
                                                double* __restrict__ stats_lsq_cum) -> void;
}  // namespace bluebild
