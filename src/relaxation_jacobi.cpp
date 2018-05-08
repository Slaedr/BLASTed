/** \file relaxation_jacobi.cpp
 * \brief Implementation of Jacobi relaxation
 * \author Aditya Kashi
 * \date 2018-04
 */

#include "relaxation_jacobi.hpp"
#include "kernels/kernels_relaxation.hpp"

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiRelaxation<scalar,index,bs,stor>::apply(const scalar *const bb, 
		scalar *const __restrict xx) const
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	Eigen::aligned_allocator<Seg> ea;
	Seg* xtemp = ea.allocate(mat.nbrows);

	const Blk *data = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *b = reinterpret_cast<const Seg*>(bb);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	scalar refdiffnorm = 1;

	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp parallel for default(shared)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_relax_kernel<scalar,index,bs,stor>(data, mat.bcolind, 
				irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				dblks[irow], b[irow], x, x, xtemp[irow]);
		}

		const scalar *xtempr = reinterpret_cast<const scalar*>(xtemp);

		if(solveparams.ctol)
		{
			scalar diffnorm = 0;
#pragma omp parallel for simd default(shared) reduction(+:diffnorm)
			for(index i = 0; i < mat.nbrows*bs; i++) 
			{
				const scalar diff = xtempr[i] - xx[i];
				diffnorm += diff*diff;
				xx[i] = xtempr[i];
			}
			diffnorm = std::sqrt(diffnorm);

			if(step == 0)
				refdiffnorm = diffnorm;

			if(diffnorm < solveparams.atol || diffnorm/refdiffnorm < solveparams.rtol ||
			   diffnorm/refdiffnorm > solveparams.dtol)
				break;
		}
		else
		{
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows*bs; i++) {
				xx[i] = xtempr[i];
			}
		}
	}

	ea.deallocate(xtemp,mat.nbrows);
}

template class BJacobiRelaxation<double,int,4,RowMajor>;
template class BJacobiRelaxation<double,int,4,ColMajor>;

template class BJacobiRelaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class BJacobiRelaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class BJacobiRelaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

template<typename scalar, typename index>
void jacobi_relax(const SolveParams<scalar>& sp,
		const CRawBSRMatrix<scalar,index>& mat, const scalar *const dblocks,
		const scalar *const bb, scalar *const __restrict xx)
{
	scalar* xtemp = new scalar[mat.nbrows];
	scalar refdiffnorm = 1;
	
	for(int step = 0; step < sp.maxits; step++)
	{
#pragma omp parallel for default(shared)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			xtemp[irow] = scalar_relax<scalar,index>(mat.vals, mat.bcolind, 
				mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				dblocks[irow], bb[irow], xx, xx);
		}

		if(sp.ctol)
		{
			scalar diffnorm = 0;
#pragma omp parallel for simd default(shared) reduction(+:diffnorm)
			for(index i = 0; i < mat.nbrows; i++) 
			{
				scalar diff = xtemp[i] - xx[i];
				diffnorm += diff*diff;
				xx[i] = xtemp[i];
			}
			diffnorm = std::sqrt(diffnorm);

			if(step == 0)
				refdiffnorm = diffnorm;

			if(diffnorm < sp.atol || diffnorm/refdiffnorm < sp.rtol ||
			   diffnorm/refdiffnorm > sp.dtol)
				break;
		}
		else
		{
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows; i++) {
				xx[i] = xtemp[i];
			}
		}
	}

	delete [] xtemp;
}

template<typename scalar, typename index>
void JacobiRelaxation<scalar,index>::apply(const scalar *const b, 
		scalar *const __restrict x) const
{
	jacobi_relax<scalar,index> ( solveparams, mat, dblocks, b, x);
}

template class JacobiRelaxation<double,int>;


}
