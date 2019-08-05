/** \file
 * \brief Asynchronous block-ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_ASYNC_BLOCKILU_FACTOR_H
#define BLASTED_ASYNC_BLOCKILU_FACTOR_H

#include "srmatrixdefs.hpp"
#include "async_initialization_decl.hpp"
#include "ilu_pattern.hpp"
#include "preconditioner_diagnostics.hpp"

namespace blasted {

/// Constructs the block-ILU0 factorization using a block variant of the Chow-Patel procedure
/// \cite ilu:chowpatel
/** There is currently no pre-scaling of the original matrix A, unlike the scalar ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 *
 * \param[in] mat The BSR matrix
 * \param[in] plist Lists of positions in the LU matrix required for the ILU computation
 * \param[in] nbuildsweeps Number of asynchronous sweeps to use for parallel builds
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in] init_type Type of initialization, \sa FactInit
 * \param[in] compute_remainder Pass 'true' for computing the ILU remainder before and after the
 *   factorization loop
 * \param[out] iluvals The ILU factorization non-zeros, accessed using the block-row pointers, 
 *   block-column indices and diagonal pointers of the original BSR matrix
 * \return ILU remainder
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
PrecInfo block_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                              const ILUPositions<index>& plist,
                              const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                              const FactInit init_type,
                              const bool compute_remainder,
                              scalar *const __restrict iluvals);

/// Computes the vector max norm of the ILU remainder A - LU restricted to the sparsity pattern of A
/** \param[in] mat The matrix A
 * \param[in] iluvals The non-zero entries of the LU factorization
 * \param[in] thread_chunk_size
 * \param[in,out] remvals Pre-allocated storage for the entries of the remainder matrix
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
scalar block_ilu0_nonlinear_res(const CRawBSRMatrix<scalar,index> *const mat,
                                const ILUPositions<index>& plist, const scalar *const iluvals,
                                const int thread_chunk_size);

}

#endif
