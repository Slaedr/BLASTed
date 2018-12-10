/** \file
 * \brief Simple thread-parallel BLAS-1 operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_BLAS1_H
#define BLASTED_BLAS1_H

namespace blasted {

/// Returns the largest entry in absolute value
template <typename scalar, typename index>
index maxnorm(const index N, const scalar *const vec);

}

#endif
