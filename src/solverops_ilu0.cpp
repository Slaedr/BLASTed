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
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::
AsyncBlockILU0_SRPreconditioner(const int nbuildswp, const int napplyswp, const int tcs,
                                const FactInit finit, const ApplyInit ainit,
                                const bool tf, const bool ta, const bool comp_rem)
	: iluvals{nullptr}, scale{nullptr}, ytemp{nullptr}, threadedfactor{tf}, threadedapply{ta},
	  rowscale{false}, nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{tcs},
	  factinittype{finit}, applyinittype{ainit}, compute_remainder{comp_rem}
{ }

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
                      const scalar *const iluvals,
                      scalar *const __restrict y_temp,
                      const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                      const ApplyInit init_type,
                      const scalar *const rr, scalar *const __restrict zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	
	const Blk *ilu = reinterpret_cast<const Blk*>(iluvals);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(y_temp);

	// No scaling like z := Sr done here

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
			  (ilu, mat->bcolind, mat->browptr[i], mat->diagind[i], r[i], i, y);
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
	default:
		;
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

	if(rowscale) {
		if(!scale)
			scale = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*bs*sizeof(scalar));
		else
			std::cout << "! AsyncBlockILU0_SRPreconditioner: scale was already allocated!\n";
	}
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	if(!iluvals)
		setup_storage();

	block_ilu0_factorize<scalar,index,bs,stor>
		(&mat, nbuildsweeps, thread_chunk_size, threadedfactor, factinittype,
		 compute_remainder, iluvals);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const r, 
                                                                  scalar *const __restrict z) const
{
	block_ilu0_apply<scalar,index,bs,stor>
		(&mat, iluvals, ytemp, napplysweeps, thread_chunk_size, threadedapply, applyinittype, r, z);
}

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::
AsyncILU0_SRPreconditioner(const int nbuildswp, const int napplyswp, const int tcs,
                           const FactInit fi, const ApplyInit ai,
                           const bool tf, const bool ta)
	: iluvals{nullptr}, scale{nullptr}, ytemp{nullptr}, threadedfactor{tf}, threadedapply{ta},
	  nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{tcs},
	  factinittype{fi}, applyinittype{ai}
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
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++) {
		za[i] = scale[i]*ra[i];
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
		;
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

	// correct z
#pragma omp parallel for simd default(shared)
	for(int i = 0; i < mat->nbrows; i++)
		za[i] = za[i]*scale[i];
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::setup_storage(const bool scaling)
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

	if(scaling) {
		if(!scale)
			scale = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(scalar));
		else
			std::cout << "! AsyncILU0: setup_storage(): Scale vector is already allocated!\n";
	}
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::compute()
{
	if(!iluvals)
		setup_storage(true);

	scalar_ilu0_factorize(&mat, nbuildsweeps, thread_chunk_size, threadedfactor,
	                      factinittype, iluvals, scale);
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const __restrict ra, 
                                              scalar *const __restrict za) const
{
	scalar_ilu0_apply(&mat, iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply,
	                  applyinittype, ra, za);
}

template <typename scalar, typename index>
RSAsyncILU0_SRPreconditioner<scalar,index>::
RSAsyncILU0_SRPreconditioner(const ReorderingScaling<scalar,index,1>& reorderscale,
                             const int nbuildsweeps, const int napplysweeps, const int tcs,
                             const FactInit finit, const ApplyInit ainit,
                             const bool threadedfactor, const bool threadedapply)
	: AsyncILU0_SRPreconditioner<scalar,index>(nbuildsweeps,napplysweeps, tcs, finit, ainit,
	                                           threadedfactor,threadedapply),
	rs{reorderscale}
{ }

template <typename scalar, typename index>
RSAsyncILU0_SRPreconditioner<scalar,index>::~RSAsyncILU0_SRPreconditioner()
{ }

template <typename scalar, typename index>
void RSAsyncILU0_SRPreconditioner<scalar,index>::compute()
{
	if(!iluvals)
		setup_storage(false);

	scalar_ilu0_factorize_noscale(&rsmat, nbuildsweeps, thread_chunk_size, threadedfactor,
	                              factinittype, iluvals);
}

template <typename scalar, typename index>
void RSAsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const x,
                                                       scalar *const __restrict y) const
{
}

// instantiations

template class AsyncILU0_SRPreconditioner<double,int>;

template class AsyncBlockILU0_SRPreconditioner<double,int,4,ColMajor>;
template class AsyncBlockILU0_SRPreconditioner<double,int,5,ColMajor>;

template class AsyncBlockILU0_SRPreconditioner<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class AsyncBlockILU0_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class AsyncBlockILU0_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

}
