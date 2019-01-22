/** \file solverops_sgs.cpp
 * \brief Implementation for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include <boost/align/aligned_alloc.hpp>
#include "solverops_sgs.hpp"
#include "kernels/kernels_sgs.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::
AsyncBlockSGS_SRPreconditioner(const int naswps, const ApplyInit apply_inittype,
                               const int threadchunksize)
	: ytemp{nullptr}, napplysweeps{naswps}, ainit{apply_inittype}, thread_chunk_size{threadchunksize}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockSGS_SRPreconditioner()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	BJacobiSRPreconditioner<scalar,index,bs,stor>::compute();
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*bs*sizeof(scalar));

#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			ytemp[i] = 0;
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const rr,
                                                                 scalar *const __restrict zz) const
{
	const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(ytemp);

	if(ainit == INIT_A_JACOBI || ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			ytemp[i] = 0;
	
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

	if(ainit == INIT_A_JACOBI)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			zz[i] = ytemp[i];
	else if(ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			zz[i] = 0;

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
AsyncSGS_SRPreconditioner<scalar,index>::AsyncSGS_SRPreconditioner(const int naswps,
                                                                   const ApplyInit apply_inittype,
                                                                   const int threadchunksize)
	: ytemp{nullptr}, napplysweeps{naswps}, ainit{apply_inittype}, thread_chunk_size{threadchunksize}
{ }

template <typename scalar, typename index>
AsyncSGS_SRPreconditioner<scalar,index>::~AsyncSGS_SRPreconditioner()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::compute()
{
	JacobiSRPreconditioner<scalar,index>::compute();
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(scalar));

#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			ytemp[i] = 0;
	}
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::apply(const scalar *const rr,
                                                        scalar *const __restrict zz) const
{
	if(ainit == INIT_A_JACOBI || ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			ytemp[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			ytemp[irow] = scalar_fgs(mat.vals, mat.bcolind, mat.browptr[irow], mat.diagind[irow],
			                         dblocks[irow], rr[irow], ytemp);
		}
	}

	if(ainit == INIT_A_JACOBI)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			zz[i] = ytemp[i];
	else if(ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			zz[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			zz[irow] = scalar_bgs(mat.vals, mat.bcolind, mat.diagind[irow], mat.browptr[irow+1],
			                      mat.vals[mat.diagind[irow]], dblocks[irow], ytemp[irow], zz);
		}
	}
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
