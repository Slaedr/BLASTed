/** \file solverops_ilu0.cpp
 * \brief Implementation for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include <boost/align/aligned_alloc.hpp>
#include "solverops_ilu0.hpp"
#include "async_ilu_factor.hpp"
#include "async_blockilu_factor.hpp"
#include "iter_blockilu_apply.hpp"
#include "iter_scalarilu_apply.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
::AsyncBlockILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
                                  const IterPrecParams params, const bool comp_rem)
	: SRPreconditioner<scalar,index>(std::move(matrix)), iluvals{nullptr}, scale{nullptr}, ytemp{nullptr},
	  buildparams{extractBuildIterParams(params)}, applyparams{extractApplyIterParams(params)},
	  compute_remainder{comp_rem}
{
}

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockILU0_SRPreconditioner()
{
	aligned_free(iluvals);
	aligned_free(ytemp);
	aligned_free(scale);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::setup_storage()
{
#ifdef DEBUG
	std::printf(" BSRMatrixView: precILUSetup(): First-time setup\n");
#endif

	// Allocate lu
	iluvals = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.browptr[mat.nbrows]*bs*bs*sizeof(scalar));

#pragma omp parallel for simd default(shared)
	for(index j = 0; j < mat.browptr[mat.nbrows]*bs*bs; j++) {
		iluvals[j] = mat.vals[j];
	}

	// intermediate array for the solve part
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*sizeof(scalar));
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
		{
			ytemp[i] = 0;
		}
	}
	else
		std::cout << "! AsyncBlockILU0_SRPreconditioner: Temp vector is already allocated!\n";

	if(buildparams.usescaling) {
		if(!scale)
			scale = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*sizeof(scalar));
		else
			std::cout << "! AsyncBlockILU0_SRPreconditioner: scale was already allocated!\n";
	}
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
PrecInfo AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	// first-time setup
	if(!iluvals) {
		setup_storage();
		plist = compute_ILU_positions_CSR_CSR(&mat);
	}

	return block_ilu0_factorize<scalar,index,bs,stor>
		(&mat, plist, buildparams.nsweeps, buildparams.thread_chunk_size, buildparams.threaded,
		 buildparams.inittype, compute_remainder, (buildparams.itertype == BLASTED_ITER_JACOBI),
		 iluvals, scale);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const r, 
                                                                  scalar *const __restrict z) const
{
	block_ilu0_apply<scalar,index,bs,stor>
		(&mat, iluvals, scale, ytemp, applyparams.nsweeps, applyparams.thread_chunk_size,
		 applyparams.threaded, applyparams.inittype, (applyparams.itertype == BLASTED_ITER_JACOBI),
		 r, z);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply_relax(const scalar *const r, 
                                                                        scalar *const __restrict z) const
{
	throw std::runtime_error("ILU relaxation not implemented!");
}

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::
AsyncILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
                           const IterPrecParams params, const bool compute_preconditioner_info)
	: SRPreconditioner<scalar,index>(std::move(matrix)), iluvals{nullptr}, scale{nullptr}, ytemp{nullptr},
	  buildparams{extractBuildIterParams(params)}, applyparams{extractApplyIterParams(params)},
	  compute_precinfo{compute_preconditioner_info}
{ }

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::~AsyncILU0_SRPreconditioner()
{
	aligned_free(iluvals);
	aligned_free(ytemp);
	aligned_free(scale);
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::setup_storage()
{
#ifdef DEBUG
	std::printf(" AsyncILU0 (scalar): First-time setup\n");
#endif

	// Allocate lu
	iluvals = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.browptr[mat.nbrows]*sizeof(scalar));
#pragma omp parallel for simd default(shared)
	for(int j = 0; j < mat.browptr[mat.nbrows]; j++) {
		iluvals[j] = mat.vals[j];
	}

	// intermediate array for the solve part
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(scalar));
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
		{
			ytemp[i] = 0;
		}
	}
	else
		std::cout << "! AsyncILU0: setup_storage(): Temp vector is already allocated!\n";

	if(buildparams.usescaling) {
		if(!scale)
			scale = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(scalar));
		else
			std::cout << "! AsyncILU0: setup_storage(): Scale vector is already allocated!\n";
	}
}

template <typename scalar, typename index>
PrecInfo AsyncILU0_SRPreconditioner<scalar,index>::compute()
{
	if(!iluvals) {
		setup_storage();
		plist = compute_ILU_positions_CSR_CSR(&mat);
	}

	return scalar_ilu0_factorize(&mat, plist, buildparams.nsweeps, buildparams.thread_chunk_size,
	                             buildparams.threaded, buildparams.inittype, compute_precinfo,
	                             (buildparams.itertype == BLASTED_ITER_JACOBI), iluvals, scale);

}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const __restrict ra, 
                                                     scalar *const __restrict za) const
{
	scalar_ilu0_apply(&mat, iluvals, scale, ytemp, applyparams.nsweeps, applyparams.thread_chunk_size,
	                  applyparams.threaded, applyparams.inittype,
	                  (applyparams.itertype == BLASTED_ITER_JACOBI), ra, za);
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::apply_relax(const scalar *const __restrict ra, 
                                                           scalar *const __restrict za) const
{
	throw std::runtime_error("ILU relaxation not implemented!");
}

template class AsyncILU0_SRPreconditioner<double,int>;

template class AsyncBlockILU0_SRPreconditioner<double,int,4,ColMajor>;
template class AsyncBlockILU0_SRPreconditioner<double,int,5,ColMajor>;

template class AsyncBlockILU0_SRPreconditioner<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class AsyncBlockILU0_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class AsyncBlockILU0_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif


template <typename scalar, typename index>
ReorderedAsyncILU0_SRPreconditioner<scalar,index>
::ReorderedAsyncILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
                                      ReorderingScaling<scalar,index,1> *const reorderscale,
                                      const IterPrecParams params)
	: AsyncILU0_SRPreconditioner<scalar,index>(std::move(matrix), params, false),
	  reord{reorderscale}
{ }

template <typename scalar, typename index>
ReorderedAsyncILU0_SRPreconditioner<scalar,index>::~ReorderedAsyncILU0_SRPreconditioner()
{
	alignedDestroyRawBSRMatrix<scalar,index>(rsmat);
}

template <typename scalar, typename index>
PrecInfo ReorderedAsyncILU0_SRPreconditioner<scalar,index>::compute()
{
	if(!iluvals) {
		setup_storage();
	}

	alignedDestroyRawBSRMatrix<scalar,index>(rsmat);
	rsmat = copyRawBSRMatrix<scalar,index,1>(mat);

	reord->compute(mat);
	reord->applyOrdering(rsmat, FORWARD);

	plist = compute_ILU_positions_CSR_CSR(reinterpret_cast<CRawBSRMatrix<scalar,index>*>(&rsmat));

	return scalar_ilu0_factorize(reinterpret_cast<CRawBSRMatrix<scalar,index>*>(&rsmat),plist,
	                             buildparams.nsweeps, buildparams.thread_chunk_size,
	                             buildparams.threaded, buildparams.inittype, compute_precinfo,
	                             (buildparams.itertype == BLASTED_ITER_JACOBI), iluvals, scale);
}

template <typename scalar, typename index>
void ReorderedAsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const ra,
                                                              scalar *const __restrict za) const
{
	// if a row-reordering has been set, create a temporary
	if(reord->isRowReordering())
	{
		scalar *const rb = (scalar*)aligned_alloc(CACHE_LINE_LEN, rsmat.nbrows*sizeof(scalar));

#pragma omp parallel for simd default(shared)
		for(index i = 0; i < rsmat.nbrows; i++)
			rb[i] = ra[i];

		reord->applyOrdering(rb, FORWARD, ROW);

		// solve triangular system
		scalar_ilu0_apply(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&rsmat),
		                  iluvals, scale, ytemp, applyparams.nsweeps, applyparams.thread_chunk_size,
		                  applyparams.threaded, applyparams.inittype,
		                  (applyparams.itertype == BLASTED_ITER_JACOBI), rb, za);

		aligned_free(rb);
	}
	else {
		// solve triangular system
		scalar_ilu0_apply(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&rsmat),
		                  iluvals, scale, ytemp, applyparams.nsweeps, applyparams.thread_chunk_size,
		                  applyparams.threaded, applyparams.inittype,
		                  (applyparams.itertype == BLASTED_ITER_JACOBI), ra, za);
	}

	// reorder solution
	reord->applyOrdering(za, FORWARD, COLUMN);
}

template <typename scalar, typename index>
void ReorderedAsyncILU0_SRPreconditioner<scalar,index>::apply_relax(const scalar *const x,
                                                                    scalar *const __restrict y) const
{
	throw std::runtime_error("ILU relaxation not implemented!");
}

template class ReorderedAsyncILU0_SRPreconditioner<double,int>;

#ifdef HAVE_MC64

template <typename scalar, typename index>
MC64_AsyncILU0_SRPreconditioner<scalar,index>::
MC64_AsyncILU0_SRPreconditioner(const int jb, const int nbuildsweeps, const int napplysweeps,
                                const int tcs, const FactInit finit, const ApplyInit ainit,
                                const bool threadedfactor, const bool threadedapply)
	: ReorderedAsyncILU0_SRPreconditioner<scalar,index>(new MC64(jb), nbuildsweeps,napplysweeps, tcs,
	                                                    finit, ainit,
	                                                    threadedfactor,threadedapply),
	job{jb}
{ }

template <typename scalar, typename index>
MC64_AsyncILU0_SRPreconditioner<scalar,index>::~MC64_AsyncILU0_SRPreconditioner()
{
	delete reord;
}

template class MC64_AsyncILU0_SRPreconditioner<double,int>;

#endif

}
