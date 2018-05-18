/** \file solverops_sgs.cpp
 * \brief Implementation for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include "solverops_sgs.hpp"
#include "kernels/kernels_sgs.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::AsyncBlockSGS_SRPreconditioner(const int naswps)
	: ytemp{nullptr}, napplysweeps{naswps}, thread_chunk_size{400}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockSGS_SRPreconditioner()
{
	delete [] ytemp;
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	BJacobiSRPreconditioner<scalar,index,bs,stor>::compute();
	if(!ytemp) {
		ytemp = new scalar[mat.nbrows*bs];
	}
}
	
/// Applies the block SGS preconditioner
/** If multiple threads are used, the iteration is asynchronous \cite async:anzt_triangular .
 * Assumes inverses of diagonal blocks have been computed and stored, 
 * and that \ref ytemp, a temporary storage space, has been allocated.
 * \param[in] mat The BSR matrix
 * \param[in] dblocks An array holding the inverse of diagonal blocks
 * \param ytemp A pre-allocated temporary vector needed for SGS
 * \param[in] napplysweeps Number of sweeps to use for asynchronous block-SGS application
 * \param[in] thread_chunk_size Batch size of allocation of work-items to thread contexts
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) application
 * \param[in] rr The input vector to apply the preconditioner to
 * \param[in,out] zz The output vector
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_sgs_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,
		scalar *const __restrict y_temp,
		const int napplysweeps,
		const int thread_chunk_size,
		const bool usethreads,
		const scalar *const rr, scalar *const __restrict zz)
{
	Eigen::Map<const Vector<scalar>> r(rr, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> ytemp(y_temp, mat->nbrows*bs);

	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat->nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat->nbrows*bs
		);

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows*bs; i++)
	{
		ytemp.data()[i] = 0;
	}
	
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			block_fgs<scalar, index, bs, Mattype>(data, mat->bcolind, irow, mat->browptr[irow], 
					mat->diagind[irow], dblks, r, ytemp);
		}
	}

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows*bs; i++)
	{
		z.data()[i] = ytemp.data()[i];
	}

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = mat->nbrows-1; irow >= 0; irow--)
		{
			block_bgs<scalar, index, bs, Mattype>(data, mat->bcolind, irow, mat->diagind[irow], 
					mat->browptr[irow+1], dblks, ytemp, z);
		}
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const rr,
                                                        scalar *const __restrict zz) const
{
	/*if(stor == RowMajor)
		block_sgs_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			&mat, dblocks, ytemp, napplysweeps,thread_chunk_size, true, rr, zz);
	else
		block_sgs_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(
		&mat, dblocks, ytemp, napplysweeps,thread_chunk_size, true, rr, zz);*/
	
	const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(ytemp);

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows*bs; i++)
	{
		ytemp[i] = 0;
	}
	
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_fgs<scalar, index, bs, stor>(mvals, mat.bcolind, irow, mat.browptr[irow], 
					mat.diagind[irow], dblks[irow], r[irow], y);
		}
	}

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows*bs; i++)
	{
		zz[i] = ytemp[i];
	}

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			block_bgs<scalar, index, bs, stor>(mvals, mat.bcolind, irow, mat.diagind[irow], 
					mat.browptr[irow+1], dblks[irow], y[irow], z);
		}
	}
}

template <typename scalar, typename index>
AsyncSGS_SRPreconditioner<scalar,index>::AsyncSGS_SRPreconditioner(const int naswps)
	: ytemp{nullptr}, napplysweeps{naswps}, thread_chunk_size{800}
{ }

template <typename scalar, typename index>
AsyncSGS_SRPreconditioner<scalar,index>::~AsyncSGS_SRPreconditioner()
{
	delete [] ytemp;
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::compute()
{
	JacobiSRPreconditioner<scalar,index>::compute();
	if(!ytemp) {
		ytemp = new scalar[mat.nbrows];
	}
}

/// Applies scalar SGS preconditioner
template <typename scalar, typename index>
inline
void scalar_sgs_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks, scalar *const __restrict ytemp,
		const int napplysweeps, const int thread_chunk_size, const bool usethreads,
		const scalar *const rr, scalar *const __restrict zz) 
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
		ytemp[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			ytemp[irow] = scalar_fgs(mat->vals, mat->bcolind, mat->browptr[irow], mat->diagind[irow],
					dblocks[irow], rr[irow], ytemp);
		}
	}

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
		zz[i] = ytemp[i];

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = mat->nbrows-1; irow >= 0; irow--)
		{
			zz[irow] = scalar_bgs(mat->vals, mat->bcolind, mat->diagind[irow], mat->browptr[irow+1],
					mat->vals[mat->diagind[irow]], dblocks[irow], ytemp[irow], zz);
		}
	}
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::apply(const scalar *const rr,
                                                        scalar *const __restrict zz) const
{
	scalar_sgs_apply(&mat, dblocks, ytemp, napplysweeps, thread_chunk_size, true, rr, zz);
}

// instantiations

template class AsyncSGS_SRPreconditioner<double,int>;

template class AsyncBlockSGS_SRPreconditioner<double,int,4,ColMajor>;
template class AsyncBlockSGS_SRPreconditioner<double,int,5,ColMajor>;

template class AsyncBlockSGS_SRPreconditioner<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class AsyncBlockSGS_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class AsyncBlockSGS_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

} // end namespace
