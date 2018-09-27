/** \file
 * \brief Asynchronous scalar ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_ASYNC_ILU_FACTOR_H
#define BLASTED_ASYNC_ILU_FACTOR_H

#include "srmatrixdefs.hpp"
#include "reorderingscaling.hpp"

namespace blasted {

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
///  Scales the matrix symmetrically so that diagonal entries become 1.
/** \param[in] mat The preconditioner as a CSR matrix
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in] factinittype Method to use for initializing the ILU factor matrix
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 * \param[in,out] scale A pre-allocated array for storage of diagonal scaling factors
 */
template <typename scalar, typename index>
void scalar_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                           const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                           const FactInit factinittype,
                           scalar *const __restrict iluvals, scalar *const __restrict scale);

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
///  Does not scale the matrix
/** Scales the matrix first, and then applies the reordering.
 * \param[in] mat The preconditioner as a CSR matrix
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 */
template <typename scalar, typename index>
void scalar_ilu0_factorize_noscale(const CRawBSRMatrix<scalar,index> *const mat,
                                   const int nbuildsweeps, const int thread_chunk_size,
                                   const bool usethreads,
                                   scalar *const __restrict iluvals);

}

#endif
