/** \file
 * \brief Matrix matrix product for sparse-row type matrices
 */

#ifndef BLASTED_BLAS_MATMAT_H
#define BLASTED_BLAS_MATMAT_H

#include "srmatrixdefs.hpp"

namespace blasted {

/// Computes the sparsity pattern of the product of L and U
/** \param lu Matrix with unit lower-triangular matrix stored in the lower part and
 *    upper triangular part stored in the upper part.
 * \return A sparse-row storage with the row pointers and column-indices computed as those of the
 *   product of L and U. The non-zero values array is not allocated.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
SRMatrixStorage<scalar, const index>
computeMatMatLUSparsityPattern(const SRMatrixStorage<const scalar,const index>&& lu);

/// Computes the non-zero values of the matrix-matrix product of L and U matrices
/** \param[in] lu Matrix with unit lower-triangular matrix stored in the lower part and
 *    upper triangular part stored in the upper part.
 * \param[in,out] prod Should have the sparsity pattern of the product pre-computed.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void computeMatMatLU(const SRMatrixStorage<const scalar,const index>&& lu,
                     SRMatrixStorage<scalar,const index>& prod);

}

#endif
