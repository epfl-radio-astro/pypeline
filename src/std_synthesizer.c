#ifdef __GNUC__
#include <complex.h>
#include <mm_malloc.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include "gemmexp.h"
#include "skabb_time.h"
#include "mkl.h"
#include <assert.h>
#include <limits.h>


// Synthesizer on 3D grid, parallelized with OMP on Nw
// Single precision
// Casting to size_t as Na * Nh * Nw quickly int overflows 
//
void synthesizer_omp_sp(const float  alpha,
                        const int    Nb_, const int Ne_, const int Na_,
                        const int    Nc_, const int Nh_, const int Nw_,
                        const float  complex* V,     // V    [Nb, Ne]      C
                        const float  complex* W,     // W    [Na, Nb]      C
                        const float* XYZ,            // XYZ  [Na, Nc]      F
                        const float* GRID,           // GRID [Nc, Nh, Nw]  F  (3D)
                        float* __restrict__ OUT)     // I    [Ne, Nh, Nw]  F  (3D)
{
    const float complex czero = 0.0f + 0.0fI;
    const float complex cone  = 1.0f + 0.0fI;

    const size_t Nb = (size_t)Nb_;
    const size_t Ne = (size_t)Ne_;
    const size_t Na = (size_t)Na_;
    const size_t Nc = (size_t)Nc_;
    const size_t Nh = (size_t)Nh_;
    const size_t Nw = (size_t)Nw_;

    const size_t Nhw = Nh * Nw;

    printf("Na = %ld, Nw = %ld, Nh = %ld, Ne = %ld\n", Na, Nw, Nh, Ne);
    float size_P  = Na * Nhw * sizeof(float complex) / 1.E9;
    float size_PW = Nb * Nhw * sizeof(float complex) / 1.E9;
    float size_E  = Ne * Nhw * sizeof(float complex) / 1.E9;
    float size_tot = size_P + size_PW + size_E;
    printf("size_P = %6.2f GB, size_PW = %6.2f GB, size_E = %6.2f => size_tot = %6.2f\n",
           size_P, size_PW, size_E, size_tot);

    float complex* P  = (float complex*) _mm_malloc(Na * Nhw * sizeof(float complex), 64);
    float complex* PW = (float complex*) _mm_malloc(Nb * Nhw * sizeof(float complex), 64);
    float complex* E  = (float complex*) _mm_malloc(Ne * Nhw * sizeof(float complex), 64);
    if (P == NULL || PW == NULL || E == NULL) {
        printf( "\nERROR: Can't allocate memory for PW\n\n");
        _mm_free(P);
        _mm_free(PW);
        _mm_free(E);
        exit(1);
    }

    char transa = 't';
	char transb = 'n';

    //tic = mysecond();
    
#pragma omp parallel
    {
        //#pragma omp for schedule(static, 100) nowait
#pragma omp for
    for (size_t i=0; i<Nw; i++) {
        //
        size_t iNh    = i * Nh;
        size_t idx_g  = Nc * iNh;
        size_t idx_p  = Na * iNh;
        size_t idx_pw = Nb * iNh;
        size_t idx_e  = Ne * iNh;
        //
        //double t = mysecond();
        sgemmexp(Na, Nh, Nc, alpha, &XYZ[0], Na, &GRID[idx_g], Nc, &P[idx_p], Na);
        //printf(" ... dgemmexp %.3f sec\n", mysecond() - t);
        //
#if 0
        cgemm(&transa, &transb, &Nb, &Nh, &Na, &cone, W, &Na, &P[idx_p],   &Na, &czero, &PW[idx_pw], &Nb);
        cgemm(&transa, &transb, &Ne, &Nh, &Nb, &cone, V, &Nb, &PW[idx_pw], &Nb, &czero, &E[idx_e],   &Ne);
#else
        //t = mysecond();
        cblas_cgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Nb, Nh, Na, &cone,
                    W, Na,
                    &P[idx_p], Na,
                    &czero,
                    &PW[idx_pw], Nb);
        cblas_cgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Ne, Nh, Nb, &cone,
                    V, Nb,
                    &PW[idx_pw], Nb,
                    &czero,
                    &E[idx_e], Ne);
        //printf(" ... cblas_zgemm x2 %.3f sec\n", mysecond() - t);
#endif
        //for (size_t j=0; j<Ne*Nh; j++) {            
        //    OUT[idx_e] = creal(E[idx_e])*creal(E[idx_e]) + cimag(E[idx_e])*cimag(E[idx_e]);
        //    idx_e++;
        //}
    }

    }

    //printf("// loop 1 %.3f sec\n", mysecond() - tic);
    //printf("P[0] = %.10f %.10fi\n", creal(P[0]), cimag(P[0]));
    //tic = mysecond();
#pragma omp parallel for
    for (size_t i=0; i<Ne*Nhw; i++) {
        OUT[i] = creal(E[i])*creal(E[i]) + cimag(E[i])*cimag(E[i]);
    }
    //printf("// loop 2 %.3f; bw = %.3f\n", mysecond() - tic, Ne*Nhw*4*8 / (mysecond() - tic) / 1.E9);

    _mm_free(P);
    _mm_free(PW);
    _mm_free(E);

    
    return;
}



// Synthesizer on 3D grid, parallelized with OMP on Nw
//
void synthesizer_omp_dp(const double alpha,
                        const int Nb_, const int Ne_, const int Na_,
                        const int Nc_, const int Nh_, const int Nw_,
                        const double complex* V,     // V    [Nb, Ne]      C
                        const double complex* W,     // W    [Na, Nb]      C
                        const double* XYZ,           // XYZ  [Na, Nc]      F
                        const double* GRID,          // GRID [Nc, Nh, Nw]  F  (3D)
                        double* __restrict__ OUT)    // I    [Ne, Nh, Nw]  F  (3D)
{
    //char   label[10];
    //double tic = 0.0;

    const double complex czero = 0.0 + 0.0I;
    const double complex cone  = 1.0 + 0.0I;

    const size_t Nb = (size_t)Nb_;
    const size_t Ne = (size_t)Ne_;
    const size_t Na = (size_t)Na_;
    const size_t Nc = (size_t)Nc_;
    const size_t Nh = (size_t)Nh_;
    const size_t Nw = (size_t)Nw_;

    const size_t Nhw = Nh * Nw;

    printf("Na = %ld, Nw = %ld, Nh = %ld, Ne = %ld\n", Na, Nw, Nh, Ne);
    double size_P  = Na * Nhw * sizeof(double complex) / 1.E9;
    double size_PW = Nb * Nhw * sizeof(double complex) / 1.E9;
    double size_E  = Ne * Nhw * sizeof(double complex) / 1.E9;
    double size_tot = size_P + size_PW + size_E;
    printf("size_P = %6.2f GB, size_PW = %6.2f GB, size_E = %6.2f => size_tot = %6.2f\n",
           size_P, size_PW, size_E, size_tot);

    double complex* P  = (double complex*) _mm_malloc(Na * Nhw * sizeof(double complex), 64);
    double complex* PW = (double complex*) _mm_malloc(Nb * Nhw * sizeof(double complex), 64);
    double complex* E  = (double complex*) _mm_malloc(Ne * Nhw * sizeof(double complex), 64);
    if (P == NULL || PW == NULL || E == NULL) {
        printf( "\nERROR: Can't allocate memory for PW\n\n");
        _mm_free(P);
        _mm_free(PW);
        _mm_free(E);
        exit(1);
    }

    char transa = 't';
	char transb = 'n';

    //tic = mysecond();
    
#pragma omp parallel
    {
        //#pragma omp for schedule(static, 100) nowait
#pragma omp for
    for (size_t i=0; i<Nw; i++) {
        //
        size_t iNh    = i * Nh;
        size_t idx_g  = Nc * iNh;
        size_t idx_p  = Na * iNh;
        size_t idx_pw = Nb * iNh;
        size_t idx_e  = Ne * iNh;
        //
        //double t = mysecond();
        dgemmexp(Na, Nh, Nc, alpha, &XYZ[0], Na, &GRID[idx_g], Nc, &P[idx_p], Na);
        //printf(" ... dgemmexp %.3f sec\n", mysecond() - t);
        //
#if 0
        zgemm(&transa, &transb, &Nb, &Nh, &Na, &cone, W, &Na, &P[idx_p],   &Na, &czero, &PW[idx_pw], &Nb);
        zgemm(&transa, &transb, &Ne, &Nh, &Nb, &cone, V, &Nb, &PW[idx_pw], &Nb, &czero, &E[idx_e],   &Ne);
#else
        //t = mysecond();
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Nb, Nh, Na, &cone,
                    W, Na,
                    &P[idx_p], Na,
                    &czero,
                    &PW[idx_pw], Nb);
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Ne, Nh, Nb, &cone,
                    V, Nb,
                    &PW[idx_pw], Nb,
                    &czero,
                    &E[idx_e], Ne);
        //printf(" ... cblas_zgemm x2 %.3f sec\n", mysecond() - t);
#endif
        //for (size_t j=0; j<Ne*Nh; j++) {            
        //    OUT[idx_e] = creal(E[idx_e])*creal(E[idx_e]) + cimag(E[idx_e])*cimag(E[idx_e]);
        //    idx_e++;
        //}       
    }

    }

    //printf("// loop 1 %.3f sec\n", mysecond() - tic);
    //printf("P[0] = %.10f %.10fi\n", creal(P[0]), cimag(P[0]));
    //tic = mysecond();
#pragma omp parallel for
    for (size_t i=0; i<Ne*Nhw; i++) {
        OUT[i] = creal(E[i])*creal(E[i]) + cimag(E[i])*cimag(E[i]);
    }
    //printf("// loop 2 %.3f; bw = %.3f\n", mysecond() - tic, Ne*Nhw*4*8 / (mysecond() - tic) / 1.E9);

    _mm_free(P);
    _mm_free(PW);
    _mm_free(E);
    
    return;
}

/*
// Synthesizer with on flattened grid
//
int synthesizer(const double alpha,
                const int Nb, const int Ne, const int Na,
                const int Nc, const int Nh, const int Nw,
                const double complex* V,     // V    [Nb, Ne]     C
                const double complex* W,     // W    [Na, Nb]     C
                const double* XYZ,           // XYZ  [Na, Nc]     F
                const double* GRID,          // GRID [Nc, Nh*Nw]  F  (flattened)
                double* __restrict__ OUT)    // I    [Ne, Nh*Nw]  F
{
    const double complex czero = 0.0 + 0.0I;
    const double complex cone  = 1.0 + 0.0I;

    const int    Nhw  = Nh * Nw;

    double complex* P  = (double complex*) _mm_malloc(Na * Nhw * sizeof(double complex), 64);
    double complex* PW = (double complex*) _mm_malloc(Nb * Nhw * sizeof(double complex), 64);
    double complex* E  = (double complex*) _mm_malloc(Ne * Nhw * sizeof(double complex), 64);
    if (P == NULL || PW == NULL || E == NULL) {
        printf( "\nERROR: Can't allocate memory for matrices\n\n");
        _mm_free(P);
        _mm_free(PW);
        _mm_free(E);
        return 1;
    }

    double t = -mysecond();
    dgemmexp(Na, Nh*Nw, Nc, alpha, XYZ, Na, GRID, Nh*Nw, P, Na);
    t += mysecond();
    printf("dgemmexp %.3f sec\n", t);

#if 0
    char transa = 't';
	char transb = 'n';
    zgemm(&transa, &transb, &Nb, &Nhw, &Na, &cone, W, &Na, P,  &Na, &czero, PW, &Nb);
    zgemm(&transa, &transb, &Ne, &Nhw, &Nb, &cone, V, &Nb, PW, &Nb, &czero, E,  &Ne);
#else
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                Nb, Nhw, Na, &cone,
                W, Na,
                P, Na,
                &czero,
                PW, Nb);
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                Ne, Nhw, Nb, &cone,
                V, Nb,
                PW, Nb,
                &czero,
                E, Ne);
#endif
    for (int i=0; i<Ne*Nhw; i++) {
        OUT[i] = creal(E[i])*creal(E[i]) + cimag(E[i])*cimag(E[i]);
    }

    _mm_free(P);
    _mm_free(PW);
    _mm_free(E);

    return 0;
}
*/
