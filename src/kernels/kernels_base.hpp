/** \file kernels_base.hpp
 * \brief Definitions for preconditioning kernels
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_BASE_H
#define BLASTED_KERNELS_BASE_H

#include <Eigen/LU>

/// Shorthand for dependent templates for Eigen segment function for vectors
#define SEG template segment
/// Shorthand for dependent templates for Eigen block function for matrices
#define BLK template block

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Map;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

}

#endif
