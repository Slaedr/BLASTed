/** \file
 * \brief Asynchronous block-ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_ASYNC_BLOCKILU_FACTOR_H
#define BLASTED_ASYNC_BLOCKILU_FACTOR_H

#include "srmatrixdefs.hpp"

namespace blasted {

/// Constructs the block-ILU0 factorization using a block variant of the Chow-Patel procedure
/// \cite ilu:chowpatel
/** There is currently no pre-scaling of the original matrix A, unlike the scalar ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 *
 * The initial values of the factorization are set such that the preconditioner is SGS at worst.
 * We set L' to (I+LD^(-1)) and U' to (D+U) so that L'U' = (D+L)D^(-1)(D+U).
 *
 * \param[in] mat The BSR matrix
 * \param[in] nbuildsweeps Number of asynchronous sweeps to use for parallel builds
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[out] iluvals The ILU factorization non-zeros, accessed using the block-row pointers, 
 *   block-column indices and diagonal pointers of the original BSR matrix
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void block_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                          const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                          scalar *const __restrict iluvals);

}

#endif
