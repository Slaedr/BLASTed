/** \file relaxation_chaotic.cpp
 * \brief Implementation of chaotic relaxation
 * \author Aditya Kashi
 * \date 2018-05
 */

#include "relaxation_chaotic.hpp"
#include "kernels/kernels_relaxation.hpp"
#include <iostream>

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
ChaoticBlockRelaxation<scalar,index,bs,stor>::ChaoticBlockRelaxation(const int tcs)
	: thread_chunk_size{tcs}
{ }
	
template<typename scalar, typename index, int bs, StorageOptions stor>
void ChaoticBlockRelaxation<scalar,index,bs,stor>::apply(const scalar *const bb, 
		scalar *const __restrict xx) const
{
	const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *b = reinterpret_cast<const Seg*>(bb);
	// the solution vector is wrapped in both a pointer to const segment and one to mutable segment
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *xmut = reinterpret_cast<Seg*>(xx);

#pragma omp parallel default(shared)
	{
		for(int step = 0; step < solveparams.maxits; step++)
		{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				block_relax_kernel<scalar,index,bs,stor>
					(mvals, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
					 dblks[irow], b[irow], x, x, xmut[irow]);
			}
		}
	}
}

template class ChaoticBlockRelaxation<double,int,4,RowMajor>;
template class ChaoticBlockRelaxation<double,int,4,ColMajor>;

template class ChaoticBlockRelaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class ChaoticBlockRelaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class ChaoticBlockRelaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

template<typename scalar, typename index>
ChaoticRelaxation<scalar,index>::ChaoticRelaxation(const int threadchunksize)
	: thread_chunk_size{threadchunksize}
{ }

template<typename scalar, typename index>
void chaotic_relax(const SolveParams<scalar>& sp,
		const CRawBSRMatrix<scalar,index>& mat, const scalar *const dblocks,
		const int thread_chunk_size,
		const scalar *const bb, scalar *const __restrict xx)
{
}

template<typename scalar, typename index>
void ChaoticRelaxation<scalar,index>::apply(const scalar *const bb, 
		scalar *const __restrict xx) const
{
#pragma omp parallel default(shared)
	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait		
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			xx[irow] = scalar_relax<scalar,index>
				           (mat.vals, mat.bcolind, 
				            mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				            dblocks[irow], bb[irow], xx, xx);
		}
	}
}

template class ChaoticRelaxation<double,int>;

}
