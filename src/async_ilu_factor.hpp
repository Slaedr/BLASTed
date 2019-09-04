/** \file
 * \brief Asynchronous scalar ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_ASYNC_ILU_FACTOR_H
#define BLASTED_ASYNC_ILU_FACTOR_H

#include "srmatrixdefs.hpp"
#include "ilu_pattern.hpp"
#include "reorderingscaling.hpp"
#include "async_initialization_decl.hpp"
#include "preconditioner_diagnostics.hpp"

namespace blasted {

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
///  Scales the matrix symmetrically so that diagonal entries become 1.
/** \param[in] mat The preconditioner as a CSR matrix
 * \param[in] plist Lists of positions in the LU matrix required for the ILU computation
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in] factinittype Method to use for initializing the ILU factor matrix
 * \param[in] compute_info Whether to compute extra information such as diagonal dominance of factors
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 * \param[in,out] scale A pre-allocated array for storage of diagonal scaling factors
 */
template <typename scalar, typename index>
PrecInfo scalar_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                               const ILUPositions<index>& plist,
                               const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                               const FactInit factinittype, const bool compute_info,
                               scalar *const __restrict iluvals, scalar *const __restrict scale);

/// Computes the vector 1-norm of the ILU0 remainder A - LU
/** Note that A is assumed to be RVC, where V are the actual matrix values stored and R and C are the
 * row and column scaling factor (diagonal) matrices resp.
 */
template <typename scalar, typename index, bool needscalerow, bool needscalecol>
scalar scalar_ilu0_nonlinear_res(const CRawBSRMatrix<scalar,index> *const mat,
                                 const ILUPositions<index>& plist,
                                 const int thread_chunk_size,
                                 const scalar *const rowscale, const scalar *const colscale,
                                 const scalar *const iluvals);

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
///  Does not scale the matrix
/** Scales the matrix first, and then applies the reordering.
 * \param[in] mat The preconditioner as a CSR matrix
 * \param[in] plist Lists of positions in the LU matrix required for the ILU computation
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in] factinittype Method to use for initializing the ILU factor matrix
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 */
template <typename scalar, typename index>
void scalar_ilu0_factorize_noscale(const CRawBSRMatrix<scalar,index> *const mat,
                                   const ILUPositions<index>& plist,
                                   const int nbuildsweeps, const int thread_chunk_size,
                                   const bool usethreads, const FactInit factinittype,
                                   scalar *const __restrict iluvals);

}

#endif
