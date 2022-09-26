#include <algorithm>
#include <complex>

#include "bluebild//config.h"
#include "gpu/kernels/gram.hpp"
#include "gpu/util/gpu_runtime.hpp"
#include "gpu/util/gpu_runtime_api.hpp"

namespace bluebild {

static __device__ __forceinline__ float calc_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ double calc_sqrt(double x) { return sqrt(x); }

// compute pi * sinc(x) = pi * (sin(a * x) / (pi * x)u
static __device__ __forceinline__ float calc_pi_sinc(float a, float x) {
  return x ? sinf(a * x) / x : float(3.14159265358979323846);
}

static __device__ __forceinline__ double calc_pi_sinc(double a, double x) {
  return x ? sin(a * x) / x : double(3.14159265358979323846);
}

template <typename T>
static __global__ void gram_kernel(int n, const T* __restrict__ x, const T* __restrict__ y,
                            const T* __restrict__ z, T wl, gpu::ComplexType<T>* g, int ldg) {
  for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < n; j += gridDim.y * blockDim.y) {
    T x1 = x[j];
    T y1 = y[j];
    T z1 = z[j];
    for (int i = j + threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
      T diffX = x1 - x[i];
      T diffY = y1 - y[i];
      T diffZ = z1 - z[i];

      T norm = calc_sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
      g[i + j * ldg] = {4 * calc_pi_sinc(wl, norm), 0};
    }
  }
}

template <typename T>
auto gram_gpu(gpu::StreamType stream, int n, const T* x, const T* y, const T* z, T wl,
              gpu::ComplexType<T>* g, int ldg) -> void {
  constexpr int maxBlocks = 65535;
  constexpr int blockSize = 16;

  dim3 block(blockSize, blockSize, 1);
  auto numBlocks = std::min<unsigned int>(maxBlocks, (n + block.x - 1) / block.x);
  dim3 grid(numBlocks, numBlocks, 1);
  gpu::launch_kernel(gram_kernel<T>, grid, block, 0, stream, n, x, y, z,
                     T(2 * 3.14159265358979323846 / wl), g, ldg);
}

template auto gram_gpu<float>(gpu::StreamType stream, int n, const float* x, const float* y,
                              const float* z, float wl, gpu::ComplexType<float>* g, int ldg)
    -> void;

template auto gram_gpu<double>(gpu::StreamType stream, int n, const double* x, const double* y,
                               const double* z, double wl, gpu::ComplexType<double>* g, int ldg)
    -> void;
}  // namespace bluebild
