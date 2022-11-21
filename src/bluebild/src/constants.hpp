#pragma once

namespace bluebild {

template <typename T>
const T ZERO     = 0.00000000000000000000;
template <typename T>
const T PI       = 3.14159265358979323846;
template <typename T>
const T CST180   = 1.80000000000000000000E02;
template <typename T>
const T CST360   = 3.60000000000000000000E02;
template <typename T>
const T CST0DOT5 = 0.50000000000000000000;

template <typename T>
struct MARLA_CST {
    const T SZ_COEFF1 =  1.34959795251974073996e-11;
    const T SZ_COEFF2 =  8.86096155697856783296e-07;
    const T SZ_COEFF3 =  1.74532925199432957214e-02;
    const T CZ_COEFF1 =  3.92582397764340914444e-14;
    const T CZ_COEFF2 = -3.86632385155548605680e-09;
    const T CZ_COEFF3 =  1.52308709893354299569e-04;
};

}  // namespace bluebild
