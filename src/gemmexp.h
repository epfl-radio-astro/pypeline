#ifndef H_SKA_DGEMMEXP
#define H_SKA_DGEMMEXP


void sgemmexp(const int M, const int N, const int K, const float alpha, const float * __restrict__ A, const int lda, const float * __restrict__ B, const int ldb, float complex* __restrict__ C, const int ldc);

void dgemmexp(const int M, const int N, const int K, const double alpha, const double * __restrict__ A, const int lda, const double * __restrict__ B, const int ldb, double complex* __restrict__ C, const int ldc);

#endif
