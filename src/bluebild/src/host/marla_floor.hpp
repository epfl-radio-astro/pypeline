#pragma once

extern "C" {
#include "floor.h"
#include "floorf.h"
}

static
inline
float marla_floor_(float a) {
    return marla_floorf(a);
}

static
inline
double marla_floor_(double a) {
    return marla_floor(a);
}
