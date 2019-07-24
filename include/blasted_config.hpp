/** \file
 * \brief Configuration settings for the library
 */

#ifndef BLASTED_CONFIG_H
#define BLASTED_CONFIG_H

#include <Eigen/Core>

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Matrix;

/// Storage type for small segments of vectors
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// The storage type to use for each small dense block in case of BSR matrices
/** \warning Eigen assumes whatever alignment is necessary for efficient vectorization.
 *    This will be a problem when the data passed in by an external library is not aligned.
 *    If that is the case, we need to change the last template param to layout|Eigen::DontAlign.
 *    In that case, the code may be somewhat slower.
 */
template <typename scalar, int bs, Eigen::StorageOptions layout>
using Block_t = Eigen::Matrix<scalar,bs,bs,layout>;

/// Type to represent a small segment of a vector
/** Note that Eigen is asked to assume no alignment.
 */
template <typename scalar, int bs>
using Segment_t = Eigen::Matrix<scalar,bs,1,Eigen::ColMajor|Eigen::DontAlign>;

}

#endif
