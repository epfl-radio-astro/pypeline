#ifdef __GNUC__
#include <complex.h>
#endif
#include <stdio.h>
#include <omp.h>
#include "skabb_time.h"
#include "skabb_macros.h"

const size_t M_BLOCK_SIZE = 10000;
const size_t N_BLOCK_SIZE = 10000;


// Especially designed for [M, K] x [K, N] with K = 3
//
void sgemmexp(const int M_, const int N_, const int K_, const float alpha, const float * __restrict__ A, const int lda, const float * __restrict__ B, const int ldb, float complex* __restrict__ C, const int ldc) {

    if (K_ != 3) {
        printf("Error. %s is designed for K = 3 whereas %d was passed)", __FILE__, K_);
        exit(1);
    }

    const size_t Kb = 3;
    const size_t M  = (size_t)M_;
    const size_t N  = (size_t)N_;

    size_t idx_c = 0;

    for (size_t ib = 0; ib < M; ib += M_BLOCK_SIZE ) {
        
        size_t Mb = min(M_BLOCK_SIZE, M - ib);

#pragma vector always
        for (size_t jb = 0; jb < N; jb += N_BLOCK_SIZE) {

            size_t Nb = min(N_BLOCK_SIZE, N - jb);

            for (size_t j = 0; j < Nb; j++) {

                const float* pB = &B[jb*3 + j*Kb];

                for (size_t i = 0; i < Mb; i = i + 1) {

                    const float *pA = &A[ib + i];

                    float a0 = *(pA + 0 * lda);
                    float a1 = *(pA + 1 * lda);
                    float a2 = *(pA + 2 * lda);

                    float b0 = *(pB + 0);
                    float b1 = *(pB + 1);
                    float b2 = *(pB + 2);
                    
                    idx_c = (j + jb) * ldc + i + ib;
                    C[idx_c] = cexpf(I*alpha*(a0*b0 + a1*b1 + a2*b2));
                }
            }
        }
    }
}


// Especially designed for [M, K] x [K, N] with K = 3
//
void dgemmexp(const int M_, const int N_, const int K_, const double alpha, const double * __restrict__ A, const int lda, const double * __restrict__ B, const int ldb, double complex* __restrict__ C, const int ldc) {

    if (K_ != 3) {
        printf("Error. %s is designed for K = 3 whereas %d was passed)", __FILE__, K_);
        exit(1);
    }

    const size_t Kb = 3;
    const size_t M  = (size_t)M_;
    const size_t N  = (size_t)N_;
    
    size_t idx_c = 0;

    for (size_t ib = 0; ib < M; ib += M_BLOCK_SIZE ) {
        
        size_t Mb = min( M_BLOCK_SIZE, M - ib );

#pragma vector always
        for (size_t jb = 0; jb < N; jb += N_BLOCK_SIZE) {
            
            size_t Nb = min(N_BLOCK_SIZE, N - jb);

            for (size_t j = 0; j < Nb; j++) {

                const double* pB = &B[jb*3 + j*Kb];

                for (size_t i = 0; i < Mb; i = i + 1) {

                    const double *pA = &A[ib + i];

                    double a0 = *(pA + 0 * lda);
                    double a1 = *(pA + 1 * lda);
                    double a2 = *(pA + 2 * lda);

                    double b0 = *(pB + 0);
                    double b1 = *(pB + 1);
                    double b2 = *(pB + 2);
                    
                    idx_c = (j + jb) * ldc + i + ib;
                    C[idx_c] = cexp(I*alpha*(a0*b0 + a1*b1 + a2*b2));
                }
            }
        }
    }
}
