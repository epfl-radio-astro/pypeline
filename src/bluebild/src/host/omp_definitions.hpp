#pragma once

#include "bluebild/config.h"

#ifdef BLUEBILD_OMP
#include <omp.h>
#define BLUEBILD_OMP_PRAGMA(content) _Pragma(content)

#else
#define BLUEBILD_OMP_PRAGMA(content)
namespace bluebild {
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
inline int omp_in_parallel() { return 0; }
inline int omp_get_nested() { return 0; }
inline int omp_get_num_procs() { return 1; }
inline int omp_get_level() { return 0; }
inline void omp_set_nested(int) {}
} // namespace bluebild
#endif
