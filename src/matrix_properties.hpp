/** \file
 * \brief Some properties of matrices such as norms etc.
 */

#ifndef BLASTED_MATRIX_PROPERTIES_H
#define BLASTED_MATRIX_PROPERTIES_H

#include "srmatrixdefs.hpp"

namespace blasted {

/// Computes the average and minimum diaginal dominance of the (block-) triangular factor matrices
/** Measurement of diagonal dominance:
 * The degree of diagonal dominance of a row is defined here as
 * \f[
 *     \beta_i := (|a_{ii}| - \sum_{j \neq i} |a_{ij}|) / |a_{ii}|.
 * \f]
 * 
 * If the min of this ratio over all rows is positive, the matrix is diagonally dominant. Note that
 * \f$ \beta_i <= 1 \forall i \f$ always.
 *
 * Note that \f$ 1 - \min{i} \beta_i  = ||M||_\infty, \f$ the infinity-norm of the iteration matrix
 * of the Jacobi iteration.
 *
 * \param mat (B)SR matrix containing the (block) upper triangular matrix in its (block) upper
 *   triangular part, and the strictly lower triangular part of the unit (block) lower triangular matrix
 *   in its strictly (block) lower triangular part. So note that the (block) lower triangular matrix
 *   is assumed to have (block) unit diagonal.
 * \return An array containing, on order,
 *   - Average diagonal dominance of the lower triangular factor
 *   - Minimum diagonal dominance of the lower triangular factor
 *   - Average diagonal dominance of the upper triangular factor
 *   - Minimum diagonal dominance of the upper triangular factor.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,4> diagonal_dominance(const SRMatrixStorage<const scalar,const index>&& mat);

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

}

#endif
