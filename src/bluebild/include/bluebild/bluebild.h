#pragma once

#include "bluebild/config.h"
#include "bluebild/errors.h"

enum BluebildProcessingUnit { BLUEBILD_PU_AUTO, BLUEBILD_PU_CPU, BLUEBILD_PU_GPU };

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BluebildProcessingUnit BluebildProcessingUnit;
/*! \endcond */
#endif  // cpp

typedef void* BluebildContext;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a context.
 *
 * @param[in] pu Processing unit to use. If BLUEBILD_PU_AUTO, GPU will be used if possible, CPU
 * otherwise.
 * @param[out] ctx Context handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_ctx_create(BluebildProcessingUnit pu, BluebildContext* ctx);

/**
 * Destroy a context.
 *
 * @param[in] ctx Context handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_ctx_destroy(BluebildContext* ctx);

/**
 * Compute the positive eigenvalues and eigenvectors of a hermitian matrix in single precision.
 * Optionally solves a general eigenvalue problem.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Order of matrix A.
 * @param[in] nEig Maximum number of eigenvalues to compute.
 * @param[in] a Hermitian matrix A. Only the lower triangle is read.
 * @param[in] lda Leading dimension of A.
 * @param[in] b Matrix B. Optional. When not null, a general eigenvalue problem is solved.
 * @param[in] ldb Leading dimension of B.
 * @param[out] nEigOut Number of positive eigenvalues found.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_eigh_s(BluebildContext ctx, int m,
                                              int nEig, const void *a, int lda,
                                              const void *b, int ldb,
                                              int *nEigOut, float *d, void *v,
                                              int ldv);

/**
 * Compute the positive eigenvalues and eigenvectors of a hermitian matrix in double precision.
 * Optionally solves a general eigenvalue problem.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Order of matrix A.
 * @param[in] nEig Maximum number of eigenvalues to compute.
 * @param[in] a Matrix A.
 * @param[in] lda Leading dimension of A.
 * @param[in] b Matrix B. Optional. When not null, a general eigenvalue problem is solved.
 * @param[in] ldb Leading dimension of B.
 * @param[out] nEigOut Number of positive eigenvalues found.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_eigh_d(BluebildContext ctx, int m, int nEig, const void* a,
                                             int lda, const void* b, int ldb, int* nEigOut,
                                             double* d, void* v, int ldv);

/**
 * fPCA decomposition and data formatting for intensity field in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] wl Wavelength for which to compute the gram matrix
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] nEig Number of requested eigenvalues.
 * @param[in] s Visibility matrix.
 * @param[in] lds Leading dimension of S.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @param[out] nCluster Number of eigenpairs to output after PCA decomposition.
 * @param[in] cluster Intensity centroids for energy-level clustering.
 * @param[out] clusterIndices Cluster indices of each eigenpair.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_intensity_field_data_s(
    BluebildContext ctx, float wl, int m, int n, int nEig, const void* s, int lds, const void* w,
    int ldw, const float* xyz, int ldxyz, float* d, void* v, int ldv, int nCluster,
    const float* cluster, int* clusterIndices);

/**
 * fPCA decomposition and data formatting for intensity field in double precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] wl Wavelength for which to compute the gram matrix
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] nEig Number of requested eigenvalues.
 * @param[in] s Visibility matrix.
 * @param[in] lds Leading dimension of S.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @param[out] nCluster Number of eigenpairs to output after PCA decomposition.
 * @param[in] cluster Intensity centroids for energy-level clustering.
 * @param[out] clusterIndices Cluster indices of each eigenpair.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_intensity_field_data_d(
    BluebildContext ctx, double wl, int m, int n, int nEig, const void* s, int lds, const void* w,
    int ldw, const double* xyz, int ldxyz, double* d, void* v, int ldv, int nCluster,
    const double* cluster, int* clusterIndices);

/**
 * Data processor for computing sensitivity fields in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] wl Wavelength for which to compute the gram matrix
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] nEig Number of requested eigenvalues.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_sensitivity_field_data_s(BluebildContext ctx, float wl,
                                                                int m, int n, int nEig, void* w,
                                                                int ldw, const float* xyz,
                                                                int ldxyz, float* d, void* v,
                                                                int ldv);

/**
 * Data processor for computing sensitivity fields in double precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] wl Wavelength for which to compute the gram matrix
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] nEig Number of requested eigenvalues.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[out] d Eigenvalues.
 * @param[out] v Eigenvectors stored as Matrix coloumns.
 * @param[out] ldv Leading of V.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_sensitivity_field_data_d(BluebildContext ctx, double wl,
                                                                int m, int n, int nEig, void* w,
                                                                int ldw, const double* xyz,
                                                                int ldxyz, double* d, void* v,
                                                                int ldv);

/**
 * Data processor for the gram matrix in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[in] wl Wavelength for which to compute the gram matrix.
 * @param[out] g Gram matrix.
 * @param[out] ldg Leading of G.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_s(BluebildContext ctx, int m, int n,
                                                     const void* w, int ldw, const float* xyz,
                                                     int ldxyz, float wl, void* g, int ldg);

/**
 * Data processor for the gram matrix in double precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] m Number of antenna.
 * @param[in] n Number of beams.
 * @param[in] w Beamforming matrix.
 * @param[in] ldw Leading dimension of W.
 * @param[in] xyz Three dimensional antenna coordinates, where each coloumn represents one
 * dimension.
 * @param[in] ldxyz Leading dimension of xyz.
 * @param[in] wl Wavelength for which to compute the gram matrix.
 * @param[out] g Gram matrix.
 * @param[out] ldg Leading of G.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_gram_matrix_d(BluebildContext ctx, int m, int n,
                                                     const void* w, int ldw, const double* xyz,
                                                     int ldxyz, double wl, void* g, int ldg);
#ifdef __cplusplus
}
#endif
