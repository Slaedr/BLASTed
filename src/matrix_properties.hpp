/** \file
 * \brief Some properties of matrices such as norms etc.
 */

#ifndef BLASTED_MATRIX_PROPERTIES_H
#define BLASTED_MATRIX_PROPERTIES_H

#include "srmatrixdefs.hpp"

namespace blasted {

/** \defgroup diagonal_dominance Measurement of diagonal dominance
 * The degree of diagonal dominance is defined here as the ratio of the diagonal entry in a row
 * to the sum of off-diagonal entries of that row. If the min of this ratio over all rows is more than 1,
 * the matrix is diagonally dominant.
 *
 * Note that the inverse of the minimum of this degree of diagonal dominance is an upper bound on the
 * infinity-norm of the iteration matrix of the Jacobi iteration.
 * @{
 */

/// Computes the average and minimum diaginal dominance of the (block-)upper triangular factor matrix
/**
 * \param mat The matrix whose upper triangular part is the U factor
 * \return The first entry is the average diagonal dominance while the second is the minimum.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,2> diagonal_dominance_upper(const SRMatrixStorage<const scalar,const index>&& mat);

/// Computes the average and minimum diaginal dominance of the (block-)lower triangular factor matrix
/** \param mat The matrix whose strictly lower triangular part is that of the L factor.
 *    L is assumed to have unit (block-)diagonal.
 * \return The first entry is the average diagonal dominance while the second is the minimum.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,2> diagonal_dominance_lower(const SRMatrixStorage<const scalar,const index>&& mat);

/** @} */

}

#endif
