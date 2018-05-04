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
	  nbuildsweeps{nbuildswp}, napplysweeps{napplyswp}, thread_chunk_size{400}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockILU0_SRPreconditioner()
{
	delete [] iluvals;
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
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_ilu0_setup(const CRawBSRMatrix<scalar,index> *const mat,
		const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
		scalar *const __restrict ilu
	)
{
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<Mattype> iluvals(ilu, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

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
					Matrix<scalar,bs,bs> sum = Mattype::IsRowMajor ?
						data.BLK<bs,bs>(j*bs,0) : data.BLK<bs,bs>(0,j*bs);

					for( index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
							k++
						) 
					{
						index pos = -1;
						inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

						if(pos == -1) continue;

						if(Mattype::IsRowMajor)
							sum.noalias() -= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
						else
							sum.noalias() -= iluvals.BLK<bs,bs>(0,k*bs)*iluvals.BLK<bs,bs>(0,pos*bs);
					}

					if(Mattype::IsRowMajor)
						iluvals.BLK<bs,bs>(j*bs,0).noalias()
							= sum * iluvals.BLK<bs,bs>(mat->diagind[mat->bcolind[j]]*bs,0).inverse();
					else
						iluvals.BLK<bs,bs>(0,j*bs).noalias()
							= sum * iluvals.BLK<bs,bs>(0,mat->diagind[mat->bcolind[j]]*bs).inverse();
				}
				else
				{
					// compute u_ij
					if(Mattype::IsRowMajor)
						iluvals.BLK<bs,bs>(j*bs,0) = data.BLK<bs,bs>(j*bs,0);
					else
						iluvals.BLK<bs,bs>(0,j*bs) = data.BLK<bs,bs>(0,j*bs);

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

						if(Mattype::IsRowMajor)
							iluvals.BLK<bs,bs>(j*bs,0).noalias()
								-= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
						else
							iluvals.BLK<bs,bs>(0,j*bs).noalias()
								-= iluvals.BLK<bs,bs>(0,k*bs)*iluvals.BLK<bs,bs>(0,pos*bs);
					}
				}
			}
		}
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		if(Mattype::IsRowMajor)
			iluvals.BLK<bs,bs>(mat->diagind[irow]*bs,0) 
				= iluvals.BLK<bs,bs>(mat->diagind[irow]*bs,0).inverse().eval();
		else
			iluvals.BLK<bs,bs>(0,mat->diagind[irow]*bs) 
				= iluvals.BLK<bs,bs>(0,mat->diagind[irow]*bs).inverse().eval();
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
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_ilu0_apply( const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const ilu,
		scalar *const __restrict y_temp,
		const int napplysweeps, const int thread_chunk_size, const bool usethreads,
		const scalar *const r, 
        scalar *const __restrict z
	)
{
	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");

	Eigen::Map<const Vector<scalar>> ra(r, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> za(z, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> ytemp(y_temp, mat->nbrows*bs);
	Eigen::Map<const Mattype> iluvals(ilu, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

	// No scaling like z := Sr done here
	
	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index i = 0; i < mat->nbrows; i++)
		{
			block_unit_lower_triangular<scalar,index,bs,Mattype>(iluvals, mat->bcolind, i, 
					mat->browptr[i], mat->diagind[i], ra, ytemp);
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
			block_upper_triangular<scalar,index,bs,Mattype>(iluvals, mat->bcolind, i, 
					mat->diagind[i], mat->browptr[i+1], ytemp, za);
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
		iluvals = new scalar[mat.browptr[mat.nbrows]*bs*bs];
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
			std::cout << "! BSRMatrix: precILUSetup(): Temp vector is already allocated!\n";
	}

	if(stor == RowMajor)
		block_ilu0_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			(&mat, nbuildsweeps, thread_chunk_size, threadedfactor, iluvals);
	else
		block_ilu0_setup<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
			(&mat, nbuildsweeps, thread_chunk_size, threadedfactor, iluvals);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const r, 
                                              scalar *const __restrict z) const
{
	if(stor == RowMajor)
		block_ilu0_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			(&mat, iluvals, ytemp, napplysweeps, thread_chunk_size, threadedapply, r, z);
	else
		block_ilu0_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
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
