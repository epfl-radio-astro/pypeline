#include <algorithm>
#include <complex>

#include "bluebild/config.h"
#include "gpu/kernels/standard_synthesizer.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

static __device__ __forceinline__
size_t get_1d_2d_thread_index() {
    return (size_t)blockIdx.x * (size_t)blockDim.x * (size_t)blockDim.y
        + (size_t)threadIdx.y * (size_t)blockDim.x + (size_t)threadIdx.x;
}

static __device__ __forceinline__
void calc_sincos(float x, float* sptr, float* cptr) {
    sincosf(x, sptr, cptr);
}

static __device__ __forceinline__
void calc_sincos(double x, double* sptr, double* cptr) {
    sincos(x, sptr, cptr);
}

static __device__ __forceinline__
double calc_norm(gpu::ComplexType<double> x) {
    return x.x * x.x + x.y * x.y;
}
static __device__ __forceinline__
float calc_norm(gpu::ComplexType<float> x) {
    return x.x * x.x + x.y * x.y;
}


template <typename T>
static __global__ void ss_p_kernel(const T alpha,
                                   const int Na, const int Nh, const int Nw,
                                   const T* __restrict__ xyz, const T* __restrict__ grid,
                                   const size_t offset, gpu::ComplexType<T>* __restrict__ p) {
    //assert(blockDim.z == 1);

    // Slice / k offset for block of P being populated
    int bk_off = offset / (Na * Nh);
    
    // Numbers of thread blocks along Na (i axis) and Nh (j axis)
    int nbi = (Na + blockDim.x - 1) / blockDim.x;
    int nbj = (Nh + blockDim.y - 1) / blockDim.y;

    // P 3D block indices
    int bk =  blockIdx.x / (nbi * nbj); // Slice along Nw (k axis)
    int bj = (blockIdx.x % (nbi * nbj)) / nbi;
    int bi =  blockIdx.x - bk * (nbi * nbj) - bj * nbi;


    if (bi * blockDim.x + threadIdx.x < Na && bj * blockDim.y + threadIdx.y < Nh) {

#ifdef K_P_SHMEM

        extern __shared__ T sh_mem[];
        T* xyz_  = &sh_mem[0];
        T* grid_ = &sh_mem[blockDim.x * 3];
        //cuDoubleComplex* cosin = (cuDoubleComplex*) &sh_mem[blockDim.x * 3 + blockDim.y * 3];

        if (threadIdx.y == 0) {
            xyz_[3 * threadIdx.x + 0] = xyz[bi * blockDim.x + threadIdx.x + 0 * Na];
            xyz_[3 * threadIdx.x + 1] = xyz[bi * blockDim.x + threadIdx.x + 1 * Na];
            xyz_[3 * threadIdx.x + 2] = xyz[bi * blockDim.x + threadIdx.x + 2 * Na];
        }

        if (threadIdx.x == 0) {
            grid_[3 * threadIdx.y + 0] = grid[(bk + bk_off) * 3 * Nh + (bj * blockDim.y + threadIdx.y) * 3 + 0];
            grid_[3 * threadIdx.y + 1] = grid[(bk + bk_off) * 3 * Nh + (bj * blockDim.y + threadIdx.y) * 3 + 1];
            grid_[3 * threadIdx.y + 2] = grid[(bk + bk_off) * 3 * Nh + (bj * blockDim.y + threadIdx.y) * 3 + 2];
        }
       
        __syncthreads();

        T a0 = xyz_[3 * threadIdx.x];
        T a1 = xyz_[3 * threadIdx.x + 1];
        T a2 = xyz_[3 * threadIdx.x + 2];
        
        T b0 = grid_[3 * threadIdx.y];
        T b1 = grid_[3 * threadIdx.y + 1];
        T b2 = grid_[3 * threadIdx.y + 2];
#else
        int a_off = bi * blockDim.x + threadIdx.x;       
        T a0 = xyz[a_off];
        T a1 = xyz[a_off + Na];
        T a2 = xyz[a_off + 2 * Na];

        int b_off = (bk + bk_off) * 3 * Nh + (bj * blockDim.y + threadIdx.y) * 3;
        T b0 = grid[b_off];
        T b1 = grid[b_off + 1];
        T b2 = grid[b_off + 2];   
#endif
        
        T tmp = alpha * (a0 * b0 + a1 * b1 + a2 * b2);
   
        size_t idx = bk * static_cast<size_t>(Na * Nh) + bj * blockDim.y * Na + threadIdx.y * Na + bi * blockDim.x + threadIdx.x;

        calc_sincos(tmp, &p[idx].y, &p[idx].x);
    }
}


template <typename T>
static __global__ void ss_stats_kernel(const T* d, const gpu::ComplexType<T>* e, const size_t Ne, const size_t Nh,
                                       const size_t Nl, const size_t Nw, const size_t* c_idx, const size_t* c_info,
                                       T* stats_std_cum, T* stats_lsq_cum) {

    size_t tid = get_1d_2d_thread_index();

    extern __shared__ size_t info[]; //starts + thicknesses
    size_t* shm_c = &info[0];
    T*      shm_d = (T*)&info[2*Nl];

    if (threadIdx.x < 2 * Nl)  shm_c[threadIdx.x] = c_info[threadIdx.x];
    if (threadIdx.x < Ne)      shm_d[threadIdx.x] = d[threadIdx.x];
    __syncthreads();

    // Each thread in charge of 1 cell
    if (tid < Nl * Nh * Nw) {
        size_t k = tid / (Nl * Nh);            // plane
        size_t j = (tid - k * Nl * Nh) / Nl;   // col
        size_t i = tid - k * Nl * Nh - j * Nl; // row

        // idx of first value to sum up for the corresponding level
        size_t idx_unlay = k * Ne * Nh + j * Ne + shm_c[i];

        T val_std = 0.0, val_lsq = 0.0;
        for (size_t l = 0; l < shm_c[Nl+i]; l++) {
            T norm_ = calc_norm(e[idx_unlay + l]);
            val_std += norm_;
            val_lsq += norm_ * shm_d[shm_c[i]+l];
        }
        stats_std_cum[tid] += val_std;
        stats_lsq_cum[tid] += val_lsq;
    }
}

template <typename T>
auto standard_synthesizer_p_gpu(gpu::StreamType stream, const T wl, const T* pix_grid,
                                const T* xyz, const size_t Na, const size_t Nh, const size_t Nw,
                                gpu::ComplexType<T>* p) -> void {

    T alpha = 2.0 * M_PI  / wl;
    const size_t SIZE2 = Nw;
    const dim3 block_p(32, 16);
    const uint nbi = (Na + block_p.x - 1) / block_p.x; // Number of blocks needeed along Na (i axis)
    const uint nbj = (Nh + block_p.y - 1) / block_p.y; // Number of blocks needeed along Nh (j axis)
    const dim3 grid_p(SIZE2 * nbi * nbj); // max is 2^31-1

    size_t nby_smem_p = 3 * (block_p.x + block_p.y) * sizeof(T);

    gpu::launch_kernel(ss_p_kernel<T>, grid_p, block_p, nby_smem_p, stream,
                       alpha, Na, Nh, SIZE2, xyz, pix_grid, 0 + 0, p);
}

template <typename T>
auto standard_synthesizer_stats_gpu(gpu::StreamType stream, const T* d, const gpu::ComplexType<T>* e,
                                    const size_t Ne, const size_t Nh, const size_t Nl, const size_t Nw,
                                    const size_t* c_idx, const size_t* c_thickness,
                                    T* stats_std_cum, T* stats_lsq_cum) -> void {

    const dim3 block_i(64, 1);
    const dim3 grid_i((Nl * Nh * Nw + block_i.x * block_i.y - 1) / (block_i.x * block_i.y));
    const size_t shm_bytes = 2 * Nl * sizeof(size_t) + Ne * sizeof(T);
    gpu::launch_kernel(ss_stats_kernel<T>, grid_i, block_i, shm_bytes, stream, d, e, Ne, Nh, Nl, Nw,
                       c_idx, c_thickness, stats_std_cum, stats_lsq_cum);
}

template auto standard_synthesizer_stats_gpu<float>(gpu::StreamType stream, const float* d, const gpu::ComplexType<float>* e,
                                                    const size_t Ne, const size_t Nh, const size_t Nl, const size_t Nw,
                                                    const size_t* c_idx, const size_t* c_thickness,
                                                    float* stats_std_cum, float* stats_lsq_cum) -> void;

template auto standard_synthesizer_stats_gpu<double>(gpu::StreamType stream, const double* d, const gpu::ComplexType<double>* e,
                                                     const size_t Ne, const size_t Nh, const size_t Nl, const size_t Nw,
                                                     const size_t* c_idx, const size_t* c_thickness,
                                                     double* stats_std_cum, double* stats_lsq_cum) -> void;

template auto standard_synthesizer_p_gpu<float>(gpu::StreamType stream, const float wl, const float* pix_grid,
                                                const float* xyz, const size_t Na, const size_t Nh, const size_t Nw,
                                                gpu::ComplexType<float>* p) -> void;

template auto standard_synthesizer_p_gpu<double>(gpu::StreamType stream, const double wl, const double* pix_grid,
                                                 const double* xyz, const size_t Na, const size_t Nh, const size_t Nw,
                                                 gpu::ComplexType<double>* p) -> void;
}  // namespace bluebild
