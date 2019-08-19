/** \file solverops_ilu0.cpp
 * \brief Implementation for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include <boost/align/aligned_alloc.hpp>
#include "solverops_ilu0.hpp"
#include "kernels/kernels_ilu_apply.hpp"
#include "async_ilu_factor.hpp"
#include "async_blockilu_factor.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
::AsyncBlockILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
                                  const int nbuildswp, const int napplyswp, const bool uscl,
                                  const int tcs, const FactInit finit, const ApplyInit ainit,
                                  const bool tf, const bool ta, const bool comp_rem)
	: SRPreconditioner<scalar,index>(std::move(matrix)), iluvals{nullptr}, scale{nullptr}, ytemp{nullptr},
	  usescaling{uscl}, threadedfactor{tf}, threadedapply{ta},
	  nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{tcs},
	  factinittype{finit}, applyinittype{ainit}, compute_remainder{comp_rem}
{
}

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockILU0_SRPreconditioner()
{
	aligned_free(iluvals);
	aligned_free(ytemp);
	aligned_free(scale);
}

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
 * \param[in] r The RHS vector of the preconditioning problem Mz = r
 * \param[in,out] z The solution vector of the preconditioning problem Mz = r
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
inline
void block_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                      const scalar *const iluvals, const scalar *const scale,
                      scalar *const __restrict y_temp,
                      const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                      const ApplyInit init_type,
                      const scalar *const rr, scalar *const __restrict zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	const Blk *ilu = reinterpret_cast<const Blk*>(iluvals);
	//const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(y_temp);

	if(scale)
		// initially, z := Sr
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++) {
			zz[i] = scale[i]*rr[i];
		}
	else
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++) {
			zz[i] = rr[i];
		}

	switch(init_type) {
	case INIT_A_JACOBI:
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			y_temp[i] = 0;
		}
		break;
	default:
		;
	}

	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index i = 0; i < mat->nbrows; i++)
		{
			block_unit_lower_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->browptr[i], mat->diagind[i], z[i], i, y);
		}
	}

	switch(init_type) {
	case INIT_A_JACOBI:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			zz[i] = y_temp[i];
		}
		break;
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			zz[i] = 0;
		}
		break;
	default:
		throw std::runtime_error(" scalar_ilu0_apply: Invalid init type!");
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			block_upper_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->diagind[i], mat->browptr[i+1], y[i], i, z);
		}
	}

	// scale z
	if(scale)
#pragma omp parallel for simd default(shared)
		for(int i = 0; i < mat->nbrows*bs; i++)
			zz[i] = zz[i]*scale[i];
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

	if(usescaling) {
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
		(&mat, plist, nbuildsweeps, thread_chunk_size, threadedfactor, factinittype,
		 compute_remainder, iluvals, scale);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const r, 
                                                                  scalar *const __restrict z) const
{
	block_ilu0_apply<scalar,index,bs,stor>
		(&mat, iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply, applyinittype, r, z);
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
                           const int nbuildswp, const int napplyswp, const bool uscal, const int tcs,
                           const FactInit fi, const ApplyInit ai, const bool compute_preconditioner_info,
                           const bool tf, const bool ta)
	: SRPreconditioner<scalar,index>(std::move(matrix)),
	  iluvals{nullptr}, scale{nullptr}, ytemp{nullptr}, usescaling{uscal},
	  threadedfactor{tf}, threadedapply{ta},
	  nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{tcs},
	  factinittype{fi}, applyinittype{ai}, compute_precinfo{compute_preconditioner_info}
{ }

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::~AsyncILU0_SRPreconditioner()
{
	aligned_free(iluvals);
	aligned_free(ytemp);
	aligned_free(scale);
}

template <typename scalar, typename index>
inline
void scalar_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                       const scalar *const iluvals, const scalar *const scale,
                       scalar *const __restrict ytemp,
                       const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                       const ApplyInit init_type,
                       const scalar *const ra, scalar *const __restrict za) 
{
	// initially, z := Sr
	if(scale)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = scale[i]*ra[i];
		}
	else
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = ra[i];
		}

	switch(init_type) {
	case INIT_A_JACOBI:
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			ytemp[i] = 0;
		}
	break;
	default:
		;
	}

	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index i = 0; i < mat->nbrows; i++)
		{
			ytemp[i] = scalar_unit_lower_triangular(iluvals, mat->bcolind, mat->browptr[i],
					mat->diagind[i], za[i], ytemp);
		}
	}

	switch(init_type) {
	case INIT_A_JACOBI:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = ytemp[i];
		}
		break;
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = 0;
		}
		break;
	default:
		throw std::runtime_error(" scalar_ilu0_apply: Invalid init type!");
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			za[i] = scalar_upper_triangular<scalar,index>(iluvals, mat->bcolind, mat->diagind[i],
					mat->browptr[i+1], 1.0/iluvals[mat->diagind[i]], ytemp[i], za);
		}
	}

	if(scale)
		// scale z
#pragma omp parallel for simd default(shared)
		for(int i = 0; i < mat->nbrows; i++)
			za[i] = za[i]*scale[i];
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

	if(usescaling) {
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

	return scalar_ilu0_factorize(&mat, plist, nbuildsweeps, thread_chunk_size, threadedfactor,
	                             factinittype, compute_precinfo, iluvals, scale);

}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const __restrict ra, 
                                                     scalar *const __restrict za) const
{
	scalar_ilu0_apply(&mat, iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply,
	                  applyinittype, ra, za);
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
                                      const int nbuildsweeps, const int napplysweeps, const int tcs,
                                      const FactInit finit, const ApplyInit ainit,
                                      const bool threadedfactor, const bool threadedapply)
	: AsyncILU0_SRPreconditioner<scalar,index>(std::move(matrix),nbuildsweeps,napplysweeps,true, tcs,
	                                           finit, ainit, threadedfactor,threadedapply),
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

	return scalar_ilu0_factorize(reinterpret_cast<CRawBSRMatrix<scalar,index>*>(&rsmat),
	                             plist, nbuildsweeps, thread_chunk_size, threadedfactor,
	                             factinittype, false, iluvals, scale);
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
		                  iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply,
		                  applyinittype, rb, za);

		aligned_free(rb);
	}
	else {
		// solve triangular system
		scalar_ilu0_apply(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&rsmat),
		                  iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply,
		                  applyinittype, ra, za);
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
