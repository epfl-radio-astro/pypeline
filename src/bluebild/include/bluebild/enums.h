#pragma once

#include "bluebild/config.h"

enum BluebildProcessingUnit { BLUEBILD_PU_AUTO, BLUEBILD_PU_CPU, BLUEBILD_PU_GPU };

enum BluebildFilter {
  BLUEBILD_FILTER_LSQ,
  BLUEBILD_FILTER_STD,
  BLUEBILD_FILTER_SQRT,
  BLUEBILD_FILTER_INV
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BluebildProcessingUnit BluebildProcessingUnit;
typedef enum BluebildFilter BluebildFilter;
/*! \endcond */
#endif  // cpp

