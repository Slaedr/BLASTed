/** \file iter_blockilu_apply.cpp
 * \brief Header for iterative triangular solvers for ILU factorizations
 * \author Aditya Kashi
 */

#ifndef BLASTED_ITER_BLOCKILU_APPLY_H
#define BLASTED_ITER_BLOCKILU_APPLY_H

#include "srmatrixdefs.hpp"
#include "async_initialization_decl.hpp"

namespace blasted {

/// Applies the block-ILU0 factorization using a block variant of the asynch triangular solve in
/// \cite async:anzt_triangular
/**
 * \param[in] mat The BSR matrix
 * \param[in] iluvals The ILU factorization non-zeros, accessed using the block-row pointers,
 *   block-column indices and diagonal pointers of the original BSR matrix
 * \param ytemp A pre-allocated temporary vector, needed for applying the ILU0 factors
 * \param[in] napplysweeps Number of asynchronous sweeps to use for parallel application
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) application
 * \param[in] init_type Type of initialization
 * \param[in] jacobiiter Set to true to use Jacobi iterations for triangular solve instead of async
 * \param[in] r The RHS vector of the preconditioning problem Mz = r
 * \param[in,out] z The solution vector of the preconditioning problem Mz = r
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void block_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                      const scalar *const iluvals, const scalar *const scale,
                      scalar *const __restrict y_temp,
                      const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                      const ApplyInit init_type, const bool jacobiiter,
                      const scalar *const rr, scalar *const __restrict zz);

}

#endif
