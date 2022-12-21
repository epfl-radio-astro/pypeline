#include <algorithm>

#include "bluebild/config.h"
#include "gpu/kernels/gemmexp.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

#include <cub/cub.cuh>

namespace bluebild {

static __device__ __forceinline__ void calc_sincos(float x, float *sptr,
                                                   float *cptr) {
  sincosf(x, sptr, cptr);
}

static __device__ __forceinline__ void calc_sincos(double x, double *sptr,
                                                   double *cptr) {
  sincos(x, sptr, cptr);
}

namespace {
template <typename T> struct ComplexOp {
  __device__ __forceinline__ ComplexOp() = default;
  __device__ __forceinline__ ComplexOp(T x_, T y_) : x(x_), y(y_) {}
  __device__ __forceinline__ ComplexOp(const gpu::ComplexType<T> &c)
      : x(c.x), y(c.y) {}

  __device__ __forceinline__ ComplexOp<T>
  operator-(const ComplexOp<T> &other) const {
    return ComplexOp{x - other.x, y - other.y};
  }

  __device__ __forceinline__ ComplexOp<T>
  operator+(const ComplexOp<T> &other) const {
    return ComplexOp{x + other.x, y + other.y};
  }

  __device__ __forceinline__ ComplexOp<T>
  operator*(const ComplexOp<T> &other) const {
    return ComplexOp{x * other.x - y * other.y, x * other.y + other.x * y};
  }

  T x, y;
};
} // namespace

template <typename T, size_t BLOCK_THREADS, cub::BlockReduceAlgorithm ALGORITHM>
static __global__ void
gemmexp_kernel(size_t nEig, size_t nPixel, size_t nAntenna, T alpha,
               const gpu::ComplexType<T> *__restrict__ vUnbeam, size_t ldv,
               const T *__restrict__ xyz, size_t ldxyz,
               const T *__restrict__ pixelX, const T *__restrict__ pixelY,
               const T *__restrict__ pixelZ, T *__restrict__ out,
               size_t ldout) {
  using BlockReduceType =
      cub::BlockReduce<ComplexOp<T>, BLOCK_THREADS, ALGORITHM>;
  __shared__ typename BlockReduceType::TempStorage tmpStorage;

  for (size_t idxEig = blockIdx.y; idxEig < nEig; idxEig += gridDim.y) {
    for (size_t idxPix = blockIdx.x; idxPix < nPixel; idxPix += gridDim.x) {
      const auto pX = pixelX[idxPix];
      const auto pY = pixelY[idxPix];
      const auto pZ = pixelZ[idxPix];

      ComplexOp<T> localSum{0, 0};
      for (size_t idxAnt = threadIdx.x; idxAnt < nAntenna;
           idxAnt += blockDim.x) {
        const auto imag = alpha * (pX * xyz[idxAnt] + pY * xyz[idxAnt + ldxyz] +
                                   pZ * xyz[idxAnt + 2 * ldxyz]);
        ComplexOp<T> sc;
        calc_sincos(imag, &(sc.y), &(sc.x));
        localSum = localSum + sc * vUnbeam[idxEig * ldv + idxAnt];
      }

      auto totalSum = BlockReduceType(tmpStorage).Sum(localSum);
      if (threadIdx.x == 0) {
        out[idxEig * ldout + idxPix] =
            totalSum.x * totalSum.x + totalSum.y * totalSum.y;
      }
    }
  }
}

template <typename T>
auto gemmexp_gpu(gpu::StreamType stream, std::size_t nEig, std::size_t nPixel,
                 std::size_t nAntenna, T alpha,
                 const gpu::ComplexType<T> *vUnbeam, std::size_t ldv,
                 const T *xyz, std::size_t ldxyz, const T *pixelX,
                 const T *pixelY, const T *pixelZ, T *out, std::size_t ldout)
    -> void {
  constexpr std::size_t blockSize = 512;
  constexpr std::size_t maxBlocks = 65535;

  dim3 block(blockSize, 1, 1);
  dim3 grid(std::min<std::size_t>(maxBlocks, nPixel),
            std::min<std::size_t>(maxBlocks, nEig), 1);

  gpu::launch_kernel(
      gemmexp_kernel<T, blockSize,
                     cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>,
      grid, block, 0, stream, nEig, nPixel, nAntenna, alpha, vUnbeam, ldv, xyz,
      ldxyz, pixelX, pixelY, pixelZ, out, ldout);
}

template auto
gemmexp_gpu<float>(gpu::StreamType stream, std::size_t nEig, std::size_t nPixel,
                   std::size_t nAntenna, float alpha,
                   const gpu::ComplexType<float> *__restrict__ vUnbeam,
                   std::size_t ldv, const float *__restrict__ xyz,
                   std::size_t ldxyz, const float *__restrict__ pixelX,
                   const float *__restrict__ pixelY,
                   const float *__restrict__ pixelZ, float *__restrict__ out,
                   std::size_t ldout) -> void;

template auto
gemmexp_gpu<double>(gpu::StreamType stream, std::size_t nEig,
                    std::size_t nPixel, std::size_t nAntenna, double alpha,
                    const gpu::ComplexType<double> *__restrict__ vUnbeam,
                    std::size_t ldv, const double *__restrict__ xyz,
                    std::size_t ldxyz, const double *__restrict__ pixelX,
                    const double *__restrict__ pixelY,
                    const double *__restrict__ pixelZ, double *__restrict__ out,
                    std::size_t ldout) -> void;
} // namespace bluebild
