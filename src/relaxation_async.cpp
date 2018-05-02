/** \file relaxation_async.cpp
 * \brief Implementation of asynchronous relaxations
 * \author Aditya Kashi
 * \date 2018-05
 */

#include "relaxation_async.hpp"
#include "kernels/kernels_relaxation.hpp"
#include <iostream>

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
ChaoticBlockRelaxation<scalar,index,bs,stor>::ChaoticBlockRelaxation()
	: thread_chunk_size{400}
{ }
	
template<typename scalar, typename index, int bs, class Mattype>
void chaotic_block_relax(const SolveParams<scalar>& sp,
                   const CRawBSRMatrix<scalar,index>& mat, const scalar *const dblocks,
                   const int thread_chunk_size,
                   const scalar *const bb, scalar *const __restrict xx)
{
	Eigen::Map<const Vector<scalar>> b(bb, mat.nbrows*bs);
	Eigen::Map<Vector<scalar>> xmut(xx, mat.nbrows*bs);
	Eigen::Map<const Vector<scalar>> x(xx, mat.nbrows*bs);
	
	Eigen::Map<const Mattype> data(mat.vals, 
			Mattype::IsRowMajor ? mat.browptr[mat.nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat.browptr[mat.nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat.nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat.nbrows*bs
		);

#pragma omp parallel default(shared)
	{
	for(int step = 0; step < sp.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_relax<scalar,index,bs,Mattype,Vector<scalar>>
				(data, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks, b, x, x, xmut);
		}
	}
	}
}

template<typename scalar, typename index, int bs, StorageOptions stor>
void ChaoticBlockRelaxation<scalar,index,bs,stor>::apply(const scalar *const b, 
		scalar *const __restrict x) const
{
	if(stor == RowMajor)
		chaotic_block_relax<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			( solveparams, mat, dblocks, thread_chunk_size, b, x);
	else
		chaotic_block_relax<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
			( solveparams, mat, dblocks, thread_chunk_size, b, x);
}

template class ChaoticBlockRelaxation<double,int,4,RowMajor>;
template class ChaoticBlockRelaxation<double,int,4,ColMajor>;

template class ChaoticBlockRelaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class ChaoticBlockRelaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class ChaoticBlockRelaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif


template<typename scalar, typename index, int bs, StorageOptions stor>
ABSGS_Relaxation<scalar,index,bs,stor>::ABSGS_Relaxation()
	: thread_chunk_size{400}
{ }
	
template<typename scalar, typename index, int bs, class Mattype>
void chaotic_blockSGS_relax(const SolveParams<scalar>& sp,
                   const CRawBSRMatrix<scalar,index>& mat, const scalar *const dblocks,
                   const int thread_chunk_size,
                   const scalar *const bb, scalar *const __restrict xx)
{
	Eigen::Map<const Vector<scalar>> b(bb, mat.nbrows*bs);
	Eigen::Map<Vector<scalar>> xmut(xx, mat.nbrows*bs);
	Eigen::Map<const Vector<scalar>> x(xx, mat.nbrows*bs);
	
	Eigen::Map<const Mattype> data(mat.vals, 
			Mattype::IsRowMajor ? mat.browptr[mat.nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat.browptr[mat.nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat.nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat.nbrows*bs
		);

#pragma omp parallel default(shared)
	{
	for(int step = 0; step < sp.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_relax<scalar,index,bs,Mattype,Vector<scalar>>
				(data, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks, b, x, x, xmut);
		}
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			block_relax<scalar,index,bs,Mattype,Vector<scalar>>
				(data, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks, b, x, x, xmut);
		}
	}
	}
}

template<typename scalar, typename index, int bs, StorageOptions stor>
void ABSGS_Relaxation<scalar,index,bs,stor>::apply(const scalar *const b, 
		scalar *const __restrict x) const
{
	if(stor == RowMajor)
		chaotic_blockSGS_relax<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			( solveparams, mat, dblocks, thread_chunk_size, b, x);
	else
		chaotic_blockSGS_relax<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
			( solveparams, mat, dblocks, thread_chunk_size, b, x);
}

template class ABSGS_Relaxation<double,int,4,RowMajor>;
template class ABSGS_Relaxation<double,int,4,ColMajor>;

template class ABSGS_Relaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class ABSGS_Relaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class ABSGS_Relaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

}
