/** \file relaxation_async_sgs.cpp
 * \brief Implementation of async symmetric Gauss-Seidel relaxations
 * \author Aditya Kashi
 * \date 2018-05
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "relaxation_async_sgs.hpp"
#include "kernels/kernels_relaxation.hpp"
#include <iostream>

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_Relaxation<scalar,index,bs,stor>::AsyncBlockSGS_Relaxation()
	: thread_chunk_size{400}
{ }

template<typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_Relaxation<scalar,index,bs,stor>::apply(const scalar *const bb, 
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
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			block_relax_kernel<scalar,index,bs,stor>
				(mvals, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks[irow], b[irow], x, x, xmut[irow]);
		}
	}
	}
}

template class AsyncBlockSGS_Relaxation<double,int,4,RowMajor>;
template class AsyncBlockSGS_Relaxation<double,int,4,ColMajor>;

template class AsyncBlockSGS_Relaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class AsyncBlockSGS_Relaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class AsyncBlockSGS_Relaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

template<typename scalar, typename index>
AsyncSGS_Relaxation<scalar,index>::AsyncSGS_Relaxation()
	: thread_chunk_size{800}
{ }

template<typename scalar, typename index>
void AsyncSGS_Relaxation<scalar,index>::apply(const scalar *const b, 
		scalar *const __restrict x) const
{
#pragma omp parallel default(shared)
	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait		
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			x[irow] = scalar_relax<scalar,index>
			             (mat.vals, mat.bcolind,
			              mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
			              dblocks[irow], b[irow], x, x);
		}
#pragma omp for schedule(dynamic, thread_chunk_size) nowait		
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			x[irow] = scalar_relax<scalar,index>
				               (mat.vals, mat.bcolind, 
				                mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				                dblocks[irow], b[irow], x, x);
		}		
	}
}

template class AsyncSGS_Relaxation<double,int>;	

}
