/** \file solverops_jacobi.cpp
 * \brief Implementation for (block-) Jacobi operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include <boost/align/aligned_alloc.hpp>
#include <Eigen/LU>
#include "solverops_jacobi.hpp"
#include "kernels/kernels_relaxation.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>
::BJacobiSRPreconditioner(SRMatrixStorage<const scalar,const index>&& matrix)
	: SRPreconditioner<scalar,index>(std::move(matrix)), dblocks{nullptr}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>::~BJacobiSRPreconditioner()
{
	boost::alignment::aligned_free(dblocks);
}
	
template <typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiSRPreconditioner<scalar,index,bs,stor>::compute()
{
	if(!dblocks) {
		dblocks = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*bs*sizeof(scalar));
#ifdef DEBUG
		std::cout << " precJacobiSetup(): Allocating.\n";
#endif
	}

	const Block_t<scalar,bs,stor>* vals = reinterpret_cast<const Block_t<scalar,bs,stor>*>(mat.vals);
	Block_t<scalar,bs,stor>* dblks = reinterpret_cast<Block_t<scalar,bs,stor>*>(dblocks);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
		dblks[irow] = vals[mat.diagind[irow]].inverse();
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiSRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const rr,
														 scalar *const __restrict zz) const
{
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		z[irow].noalias() = dblks[irow] * r[irow];
	}
}

template<typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiSRPreconditioner<scalar,index,bs,stor>::apply_relax(const scalar *const bb, 
                                                                scalar *const __restrict xx) const
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	scalar *xtempr = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*sizeof(scalar));

	const Blk *data = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *b = reinterpret_cast<const Seg*>(bb);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *xtemp = reinterpret_cast<Seg*>(xtempr);
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

	//ea.deallocate(xtemp,mat.nbrows);
	aligned_free(xtempr);
}

template <typename scalar, typename index>
JacobiSRPreconditioner<scalar,index>
::JacobiSRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix)
	: SRPreconditioner<scalar,index>(std::move(matrix)), dblocks{nullptr}
{ }

template <typename scalar, typename index>
JacobiSRPreconditioner<scalar,index>::~JacobiSRPreconditioner()
{
	boost::alignment::aligned_free(dblocks);
}
	
/// Inverts diagonal entries
/** \param[in] mat The matrix
 * \param[in,out] dblocks It must be pre-allocated; contains inverse of diagonal entries on exit
 */
template <typename scalar, typename index>
static inline
void scalar_jacobi_setup(const CRawBSRMatrix<scalar,index> *const mat,
                         scalar *const dblocks)
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		dblocks[irow] = 1.0/mat->vals[mat->diagind[irow]];
}

template <typename scalar, typename index>
void JacobiSRPreconditioner<scalar,index>::compute()
{
	if(!dblocks) {
		dblocks = (scalar*)boost::alignment::aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(scalar));
#ifdef DEBUG
		std::cout << " CSR MatrixView: precJacobiSetup(): Initial setup.\n";
#endif
	}

	scalar_jacobi_setup(&mat, dblocks);
}	

template <typename scalar, typename index>
void JacobiSRPreconditioner<scalar,index>::apply(const scalar *const rr,
														 scalar *const __restrict zz) const
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
		zz[irow] = dblocks[irow] * rr[irow];
}

template<typename scalar, typename index>
void JacobiSRPreconditioner<scalar,index>::apply_relax(const scalar *const bb, 
                                                       scalar *const __restrict xx) const
{
	scalar *xtemp = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(scalar));
	scalar refdiffnorm = 1;
	
	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp parallel for default(shared)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			xtemp[irow] = scalar_relax<scalar,index>(mat.vals, mat.bcolind, 
			                                         mat.browptr[irow], mat.diagind[irow],
			                                         mat.browptr[irow+1],
			                                         dblocks[irow], bb[irow], xx, xx);
		}

		if(solveparams.ctol)
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

			if(diffnorm < solveparams.atol || diffnorm/refdiffnorm < solveparams.rtol ||
			   diffnorm/refdiffnorm > solveparams.dtol)
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

	aligned_free(xtemp);
}


template class JacobiSRPreconditioner<double,int>;

template class BJacobiSRPreconditioner<double,int,4,ColMajor>;
template class BJacobiSRPreconditioner<double,int,5,ColMajor>;

template class BJacobiSRPreconditioner<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class BJacobiSRPreconditioner<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class BJacobiSRPreconditioner<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

}
