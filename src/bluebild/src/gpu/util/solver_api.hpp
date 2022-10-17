#pragma once

#include "bluebild/config.h"
#include "gpu/util/gpu_runtime_api.hpp"
#include "context_internal.hpp"

namespace bluebild {
namespace gpu {
namespace eigensolver {
auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<float> *a, int lda, float vl, float vu, int il,
           int iu, int *m, float *w) -> void;

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<double> *a, int lda, double vl, double vu, int il,
           int iu, int *m, double *w) -> void;

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<float> *a, int lda, gpu::ComplexType<float> *b,
           int ldb, float vl, float vu, int il, int iu, int *m, float *w)
    -> void;

auto solve(ContextInternal &ctx, char jobz, char range, char uplo, int n,
           gpu::ComplexType<double> *a, int lda, gpu::ComplexType<double> *b,
           int ldb, double vl, double vu, int il, int iu, int *m, double *w)
    -> void;
} // namespace eigensolver

} // namespace gpu
} // namespace bluebild
