/** \file impldefs.hpp
 * \brief Definitions needed in implementations
 * \author Aditya Kashi
 */

#ifndef BLASTED_IMPLDEFS_H
#define BLASTED_IMPLDEFS_H

#include <Eigen/Core>

namespace blasted {

/// The storage type to use for each small dense block in case of BSR matrices
template <typename scalar, int bs, Eigen::StorageOptions layout>
using Block_t = Eigen::Matrix<scalar,bs,bs,layout|Eigen::DontAlign>;

/// Type to represent a small segment of a vector
template <typename scalar, int bs>
using Segment_t = Eigen::Matrix<scalar,bs,1,Eigen::ColMajor|Eigen::DontAlign>;

}

#endif
