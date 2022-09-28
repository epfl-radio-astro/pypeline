#pragma once

#include "bluebild/config.h"
#include "bluebild/errors.h"
#include "bluebild/enums.h"

typedef void* BluebildContext;
typedef void* BluebildNufft3d3;
typedef void* BluebildNufft3d3f;

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

/**
 * Data processor for computing virtual visibilities in double precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] nFilter Number of filter.
 * @param[in] filter Array of filters.
 * @param[in] nIntervals Number of intervals.
 * @param[in] intervals Array of intervals of size (2, nIntervals).
 * @param[in] ldIntervals Leading dimension of intervals.
 * @param[in] nEig Number of eigenvalues.
 * @param[in] D Array of eignenvalues.
 * @param[in] nAntenna Number of antenna.
 * @param[in] V Visibilities of size (nBeam, nEig) if W is provided, (nAntenna, nEig) otherwise.
 * @param[in] nBeam Number of beams.
 * @param[in] W Optional beamforming matrix of size (nAntenna, nBeam). May be null.
 * @param[in] ldw Leading dimension of W.
 * @param[out] virtVis The computed virtual visibilities of size (nAntenna, nAntenna, nIntervals, nFilter).
 * @param[in] ldVirtVis1 Outer most stride.
 * @param[in] ldVirtVis2 Stride between each inner 2d matrix.
 * @param[in] ldVirtVis3 Stride between each coloumn in inner 2d matrix.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_virtual_vis_s(
    BluebildContext ctx, int nFilter, const BluebildFilter *filter,
    int nIntervals, const float *intervals, int ldIntervals, int nEig,
    const float *D, int nAntenna, const void *V, int ldv, int nBeam,
    const void *W, int ldw, void *virtVis, int ldVirtVis1, int ldVirtVis2,
    int ldVirtVis3);

/**
 * Data processor for computing virtual visibilities in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] nFilter Number of filter.
 * @param[in] filter Array of filters.
 * @param[in] nIntervals Number of intervals.
 * @param[in] intervals Array of intervals of size (2, nIntervals).
 * @param[in] ldIntervals Leading dimension of intervals.
 * @param[in] nEig Number of eigenvalues.
 * @param[in] D Array of eignenvalues.
 * @param[in] nAntenna Number of antenna.
 * @param[in] V Visibilities of size (nBeam, nEig) if W is provided, (nAntenna, nEig) otherwise.
 * @param[in] nBeam Number of beams.
 * @param[in] W Optional beamforming matrix of size (nAntenna, nBeam). May be null.
 * @param[in] ldw Leading dimension of W.
 * @param[out] virtVis The computed virtual visibilities of size (nAntenna, nAntenna, nIntervals, nFilter).
 * @param[in] ldVirtVis1 Outer most stride.
 * @param[in] ldVirtVis2 Stride between each inner 2d matrix.
 * @param[in] ldVirtVis3 Stride between each coloumn in inner 2d matrix.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_virtual_vis_d(
    BluebildContext ctx, int nFilter, const BluebildFilter *filter,
    int nIntervals, const double *intervals, int ldIntervals, int nEig,
    const double *D, int nAntenna, const void *V, int ldv, int nBeam,
    const void *W, int ldw, void *virtVis, int ldVirtVis1, int ldVirtVis2,
    int ldVirtVis3);

/**
 * Create plan for a type 3 nufft transform in 3D in single precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] iflag Sign in exponential. Either +1 or -1.
 * @param[in] tol Target preceision tolorance.
 * @param[in] numTrans Number of transforms to compute together.
 * @param[in] M Number of input points.
 * @param[in] x Input coordinates in x.
 * @param[in] y Input coordinates in y.
 * @param[in] z Input coordinates in z.
 * @param[in] N Number of output points.
 * @param[in] s Input coordinates in s.
 * @param[in] t Input coordinates in t.
 * @param[in] u Input coordinates in u.
 * @param[out] plan plan handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_create_s(
    BluebildContext ctx, int iflag, float tol, int numTrans, int M,
    const float *x, const float *y, const float *z, int N, const float *s,
    const float *t, const float *u, BluebildNufft3d3f *plan);

/**
 * Create plan for a type 3 nufft transform in 3D in double precision.
 *
 * @param[in] ctx Context handle.
 * @param[in] iflag Sign in exponential. Either +1 or -1.
 * @param[in] tol Target preceision tolorance.
 * @param[in] numTrans Number of transforms to compute together.
 * @param[in] M Number of input points.
 * @param[in] x Input coordinates in x.
 * @param[in] y Input coordinates in y.
 * @param[in] z Input coordinates in z.
 * @param[in] N Number of output points.
 * @param[in] s Input coordinates in s.
 * @param[in] t Input coordinates in t.
 * @param[in] u Input coordinates in u.
 * @param[out] plan plan handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_create_d(
    BluebildContext ctx, int iflag, double tol, int numTrans, int M,
    const double *x, const double *y, const double *z, int N, const double *s,
    const double *t, const double *u, BluebildNufft3d3* plan);

/**
 * Destroy a nufft plan handle.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError
bluebild_nufft3d3_destroy_s(BluebildNufft3d3f *plan);

/**
 * Destroy a nufft plan handle.
 *
 * @param[in] plan Plan handle.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError
bluebild_nufft3d3_destroy_d(BluebildNufft3d3 *plan);

/**
 * Execute a nufft3d3 plan in single precision.
 *
 * @param[in] plan Plan handle.
 * @param[in] cj Input data.
 * @param[out] fk Output data.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError
bluebild_nufft3d3_execute_s(BluebildNufft3d3f plan, const void *cj, void *fk);

/**
 * Execute a nufft3d3 plan in double precision.
 *
 * @param[in] plan Plan handle.
 * @param[in] cj Input data.
 * @param[out] fk Output data.
 * @return Error code or BLUEBILD_SUCCESS.
 */
BLUEBILD_EXPORT BluebildError bluebild_nufft3d3_execute_d(BluebildNufft3d3 plan,
                                                          const void *cj,
                                                          void *fk);



#ifdef __cplusplus
}
#endif
