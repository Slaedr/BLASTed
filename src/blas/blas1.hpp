/** \file
 * \brief Simple thread-parallel BLAS-1 operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_BLAS1_H
#define BLASTED_BLAS1_H

#include "device_container.hpp"

namespace blasted {

/// Returns the largest entry in absolute value
template <typename scalar>
scalar maxnorm(const device_vector<scalar>& vec);

}

#endif
