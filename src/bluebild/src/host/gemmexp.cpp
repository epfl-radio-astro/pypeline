#include "host/gemmexp.hpp"
#include "marla_sincos.hpp"
#include <iostream>

namespace bluebild {

const std::size_t M_BLOCK_SIZE = 10000;
const std::size_t N_BLOCK_SIZE = 10000;

/*
*  Special gemm with vectorized exponentiation
*/
template <typename T>
auto gemmexp(const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const T           alpha,
             const T* __restrict__ A,
             const std::size_t lda,
             const T* __restrict__ B,
             const std::size_t ldb,
             std::complex<T>* __restrict__ C,
             const std::size_t ldc) -> void {

    assert(K == 3);

    T sin_, cos_;

    const T zero = 0.0;

    std::size_t idx_c = 0;
    std::size_t idx_b = 0;
    for (std::size_t ib = 0; ib < M; ib += M_BLOCK_SIZE ) {
        std::size_t Mb = std::min(M_BLOCK_SIZE, M - ib);
        for (std::size_t jb = 0; jb < N; jb += N_BLOCK_SIZE) {
            std::size_t Nb = std::min(N_BLOCK_SIZE, N - jb);
            for (std::size_t j = 0; j < Nb; j++) {
                idx_b = jb*3 + j*K;
                idx_c = (j + jb) * ldc + ib;
#pragma vector always
                for (std::size_t i = 0; i < Mb; i = i + 1) {
                    T a0 = A[ib + i];
                    T a1 = A[ib + i + lda];
                    T a2 = A[ib + i + 2 * lda];
                    T b0 = B[idx_b];
                    T b1 = B[idx_b + 1];
                    T b2 = B[idx_b + 2];
                    T im_part = alpha * (a0*b0 + a1*b1 + a2*b2);

#ifndef __INTEL_COMPILER
                    marla_sincos(im_part, &sin_, &cos_);
                    C[idx_c + i] = std::complex<T>(cos_, sin_);
#else
                    std::complex<T> cim_part(zero, im_part);
                    C[idx_c + i] = std::exp(cim_part);
#endif                    
                }
            }
        }
    }
}

template auto gemmexp<float>(const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const float           alpha,
             const float* __restrict__ A,
             const std::size_t lda,
             const float* __restrict__ B,
             const std::size_t ldb,
             std::complex<float>* __restrict__ C,
             const std::size_t ldc) -> void;

template auto gemmexp<double>(const std::size_t M,
             const std::size_t N,
             const std::size_t K,
             const double           alpha,
             const double* __restrict__ A,
             const std::size_t lda,
             const double* __restrict__ B,
             const std::size_t ldb,
             std::complex<double>* __restrict__ C,
             const std::size_t ldc) -> void;

}
