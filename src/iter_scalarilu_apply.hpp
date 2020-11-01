/** \file iter_scalarilu_apply.hpp
 * \brief Iterative triangular solve for ILU factorizations
 */

#ifndef BLASTED_ITER_SCALARILU_APPLY_H
#define BLASTED_ITER_SCALARILU_APPLY_H

#include "srmatrixdefs.hpp"
#include "async_initialization_decl.hpp"

namespace blasted {

template <typename scalar, typename index>
void scalar_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                       const scalar *const iluvals, const scalar *const scale,
                       scalar *const __restrict ytemp,
                       const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                       const ApplyInit init_type, const bool jacobiiter,
                       const scalar *const ra, scalar *const __restrict za);

}

#endif
