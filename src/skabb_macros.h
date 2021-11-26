#ifndef H_SKABB_MACROS
#define H_SKABB_MACROS


#define PREFETCH(A)  _mm_prefetch(A, _MM_HINT_NTA)
#define PREFETCH0(A) _mm_prefetch(A, _MM_HINT_T0)
#define PREFETCH1(A) _mm_prefetch(A, _MM_HINT_T1)
#define PREFETCH2(A) _mm_prefetch(A, _MM_HINT_T2)

#define min(a,b) (((a)<(b))?(a):(b))

#define STORE128(A, B) _mm_store_pd(A, B)
#define STORE256(A, B) _mm256_store_pd(A, B)


#endif
