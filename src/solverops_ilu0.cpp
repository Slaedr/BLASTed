/** \file solverops_ilu0.cpp
 * \brief Implementation for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include "solverops_ilu0.hpp"
#include "kernels/kernels_ilu0.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::AsyncBlockILU0_SRPreconditioner
	(const int nbuildswp, const int napplyswp, const bool tf, const bool ta)
	: iluvals{nullptr}, scale{nullptr}, ytemp{nullptr}, threadedfactor{tf}, threadedapply{ta},
	  rowscale{false}, nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{400}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockILU0_SRPreconditioner()
{
	Eigen::aligned_allocator<scalar> alloc;
	alloc.deallocate(iluvals,0);
	delete [] ytemp;
	delete [] scale;
}

/// Search through inner indices
/** Finds the position in the index arary that has value indtofind
 * Searches between positions
 * \param[in] start, and
 * \param[in] end
 */
template <typename index>
static inline void inner_search(const index *const aind, 
		const index start, const index end, 
		const index indtofind, index *const pos)
{
	for(index j = start; j < end; j++) {
		if(aind[j] == indtofind) {
			*pos = j;
			break;
		}
	}
}

/// Constructs the block-ILU0 factorization using a block variant of the Chow-Patel procedure
/// \cite ilu:chowpatel
/** There is currently no pre-scaling of the original matrix A, unlike the scalar ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
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
inline
void block_ilu0_setup(const CRawBSRMatrix<scalar,index> *const mat,
		const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
		scalar *const __restrict iluvals
	)
{
	using Blk = Block_t<scalar,bs,stor>;
	
	const Blk *mvals = reinterpret_cast<const Blk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
			{
				if(irow > mat->bcolind[j])
				{
					Matrix<scalar,bs,bs> sum = mvals[j];

					for( index k = mat->browptr[irow]; 
						 (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
						 k++
					   ) 
					{
						index pos = -1;
						inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

						if(pos == -1) continue;

						sum.noalias() -= ilu[k]*ilu[pos];
					}

					ilu[j].noalias() = sum * ilu[mat->diagind[mat->bcolind[j]]].inverse();
				}
				else
				{
					// compute u_ij
					ilu[j] = mvals[j];

					for(index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index mat->bcolind[j],
						 * between the diagonal index of row mat->bcolind[k] 
						 * and the last index of row bcolind[k]
						 */
						inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

						if(pos == -1) continue;

						ilu[j].noalias() -= ilu[k]*ilu[pos];
					}
				}
			}
		}
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		ilu[mat->diagind[irow]] = ilu[mat->diagind[irow]].inverse().eval();
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
 * \param[in] r The RHS vector of the preconditioning problem Mz = r
 * \param[in,out] z The solution vector of the preconditioning problem Mz = r
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
inline
void block_ilu0_apply( const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const iluvals,
		scalar *const __restrict y_temp,
		const int napplysweeps, const int thread_chunk_size, const bool usethreads,
		const scalar *const rr, 
        scalar *const __restrict zz
	)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	
	const Blk *ilu = reinterpret_cast<const Blk*>(iluvals);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(y_temp);

	// No scaling like z := Sr done here
	
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

	// No correction of z needed because no scaling
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	if(!iluvals)
	{
#ifdef DEBUG
		std::printf(" BSRMatrixView: precILUSetup(): First-time setup\n");
#endif

		// Allocate lu
		Eigen::aligned_allocator<scalar> alloc;
		iluvals = alloc.allocate(mat.browptr[mat.nbrows]*bs*bs);
#pragma omp parallel for simd default(shared)
		for(index j = 0; j < mat.browptr[mat.nbrows]*bs*bs; j++) {
			iluvals[j] = mat.vals[j];
		}

		// intermediate array for the solve part
		if(!ytemp) {
			ytemp = new scalar[mat.nbrows*bs];
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
				scale = new scalar[mat.nbrows*bs*bs];
			else
				std::cout << "! AsyncBlockILU0_SRPreconditioner: scale was already allocated!\n";
		}
	}

	block_ilu0_setup<scalar,index,bs,stor>
	  (&mat, nbuildsweeps, thread_chunk_size, threadedfactor, iluvals);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const r, 
                                              scalar *const __restrict z) const
{
	block_ilu0_apply<scalar,index,bs,stor>
	  (&mat, iluvals, ytemp, napplysweeps, thread_chunk_size, threadedapply, r, z);
}

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::AsyncILU0_SRPreconditioner
	(const int nbuildswp, const int napplyswp, const bool tf, const bool ta)
	: iluvals{nullptr}, scale{nullptr}, ytemp{nullptr}, threadedfactor{tf}, threadedapply{ta},
	  nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{800}
{ }

template <typename scalar, typename index>
AsyncILU0_SRPreconditioner<scalar,index>::~AsyncILU0_SRPreconditioner()
{
	delete [] iluvals;
	delete [] ytemp;
	delete [] scale;
}

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
/** \param[in] mat The preconditioner as a CSR matrix
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) factorization
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 * \param[in,out] scale A pre-allocated array for storage of diagonal scaling factors
 */
template <typename scalar, typename index>
inline
void scalar_ilu0_setup(const CRawBSRMatrix<scalar,index> *const mat,
		const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
		scalar *const __restrict iluvals, scalar *const __restrict scale)
{
	// get the diagonal scaling matrix
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
		scale[i] = 1.0/std::sqrt(mat->vals[mat->diagind[i]]);

	/** initial guess
	 * We choose the initial guess such that the preconditioner reduces to SGS in the worst case.
	 */
/*#pragma omp parallel for simd default (shared)
	for(index i = 0; i < mat->browptr[mat->nbrows]; i++)
		iluvals[i] = mat->vals[i];
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++)
		for(index j = mat->browptr[i]; j < mat->diagind[j]; j++)
			iluvals[j] *= mat->vals[mat->diagind[mat->bcolind[j]]];*/

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
			{
				if(irow > mat->bcolind[j])
				{
					scalar sum = scale[irow] * mat->vals[j] * scale[mat->bcolind[j]];

					for(index k = mat->browptr[irow]; 
					    (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
					    k++  ) 
					{
						index pos = -1;
						inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

						if(pos == -1) {
							continue;
						}

						sum -= iluvals[k]*iluvals[pos];
					}

					iluvals[j] = sum / iluvals[mat->diagind[mat->bcolind[j]]];
				}
				else
				{
					// compute u_ij
					iluvals[j] = scale[irow]*mat->vals[j]*scale[mat->bcolind[j]];

					for(index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index mat->bcolind[j], 
						 * between the diagonal index of row mat->bcolind[k] 
						 * and the last index of row mat->bcolind[k]
						 */
						inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

						if(pos == -1) continue;

						iluvals[j] -= iluvals[k]*iluvals[pos];
					}
				}
			}
		}
	}
}

template <typename scalar, typename index>
inline
void scalar_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const iluvals, const scalar *const scale,
		scalar *const __restrict ytemp,
		const int napplysweeps, const int thread_chunk_size, const bool usethreads,
		const scalar *const ra, scalar *const __restrict za) 
{
	// initially, z := Sr
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++) {
		za[i] = scale[i]*ra[i];
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
void AsyncILU0_SRPreconditioner<scalar,index>::compute()
{
	if(!iluvals)
	{
#ifdef DEBUG
		std::printf(" AsyncILU0 (scalar): First-time setup\n");
#endif

		// Allocate lu
		iluvals = new scalar[mat.browptr[mat.nbrows]];
#pragma omp parallel for simd default(shared)
		for(int j = 0; j < mat.browptr[mat.nbrows]; j++) {
			iluvals[j] = mat.vals[j];
		}

		// intermediate array for the solve part
		if(!ytemp) {
			ytemp = new scalar[mat.nbrows];
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows; i++)
			{
				ytemp[i] = 0;
			}
		}
		else
			std::cout << "! BSRMatrixView<1>: precILUSetup(): Temp vector is already allocated!\n";
		
		if(!scale)
			scale = new scalar[mat.nbrows];	
		else
			std::cout << "! BSRMatrixView<1>: precILUSetup(): Scale vector is already allocated!\n";
	}

	scalar_ilu0_setup(&mat, nbuildsweeps, thread_chunk_size, threadedfactor, iluvals, scale);
}

template <typename scalar, typename index>
void AsyncILU0_SRPreconditioner<scalar,index>::apply(const scalar *const __restrict ra, 
                                              scalar *const __restrict za) const
{
	scalar_ilu0_apply(&mat, iluvals, scale, ytemp, napplysweeps, thread_chunk_size, threadedapply,
	                  ra, za);
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
