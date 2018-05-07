/** \file solverops_jacobi.cpp
 * \brief Implementation for (block-) Jacobi operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include "solverops_jacobi.hpp"
#include "kernels/kernels_base.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>::BJacobiSRPreconditioner()
	: dblocks{nullptr}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>::~BJacobiSRPreconditioner()
{
	//delete [] dblocks;
	Eigen::aligned_allocator<scalar> aa;
	aa.deallocate(dblocks, mat.nbrows*bs*bs);
}
	
/// Computes and stores the inverses of diagonal blocks
/** \ref dblocks needs to be pre-allocated.
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_jacobi_setup(const CRawBSRMatrix<scalar,index> *const mat,
						scalar *const dblocks )
{
	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat->nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat->nbrows*bs
		);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		if(Mattype::IsRowMajor)
			dblks.BLK<bs,bs>(irow*bs,0) = data.BLK<bs,bs>(mat->diagind[irow]*bs,0).inverse();
		else
			dblks.BLK<bs,bs>(0,irow*bs) = data.BLK<bs,bs>(0,mat->diagind[irow]*bs).inverse();
}

/// Applies the block-Jacobi preconditioner assuming inverses of diagonal blocks have been computed 
template <typename scalar, typename index, int bs, class Mattype>
void block_jacobi_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,
		const scalar *const rr, scalar *const __restrict zz)
{
	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		Eigen::Map<const Vector<scalar>> r(rr, mat->nbrows*bs);
		Eigen::Map<Vector<scalar>> z(zz, mat->nbrows*bs);
		Eigen::Map<const Mattype> data(mat->vals, 
				Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
				Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
			);
		Eigen::Map<const Mattype> dblks(dblocks, 
				Mattype::IsRowMajor ? mat->nbrows*bs : bs,
				Mattype::IsRowMajor ? bs : mat->nbrows*bs
			);

		if(Mattype::IsRowMajor)
			z.SEG<bs>(irow*bs).noalias() = dblks.BLK<bs,bs>(irow*bs,0) * r.SEG<bs>(irow*bs);
		else
			z.SEG<bs>(irow*bs).noalias() = dblks.BLK<bs,bs>(0,irow*bs) * r.SEG<bs>(irow*bs);
	}
}	

template <typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiSRPreconditioner<scalar,index,bs,stor>::compute()
{
	if(!dblocks) {
		Eigen::aligned_allocator<scalar> aa;
		//dblocks = new scalar[mat.nbrows*bs*bs];
		dblocks = aa.allocate(mat.nbrows*bs*bs);
#ifdef DEBUG
		std::cout << " precJacobiSetup(): Allocating.\n";
#endif
	}
	
	// if(stor == RowMajor)
	// 	block_jacobi_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(&mat, dblocks);
	// else
	// 	block_jacobi_setup<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(&mat, dblocks);

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
	if(stor == RowMajor)
		block_jacobi_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			( &mat, dblocks, rr, zz);
	else
		block_jacobi_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
			( &mat, dblocks, rr, zz);
}

template <typename scalar, typename index>
JacobiSRPreconditioner<scalar,index>::JacobiSRPreconditioner()
	: dblocks{nullptr}
{ }

template <typename scalar, typename index>
JacobiSRPreconditioner<scalar,index>::~JacobiSRPreconditioner()
{
	delete [] dblocks;
}
	
/// Inverts diagonal entries
/** \param[in] mat The matrix
 * \param[in,out] dblocks It must be pre-allocated; contains inverse of diagonal entries on exit
 */
template <typename scalar, typename index>
inline
void scalar_jacobi_setup(const CRawBSRMatrix<scalar,index> *const mat,
		scalar *const dblocks)
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		dblocks[irow] = 1.0/mat->vals[mat->diagind[irow]];
}

template <typename scalar, typename index>
inline
void scalar_jacobi_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,	
		const scalar *const rr, scalar *const __restrict zz)
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		zz[irow] = dblocks[irow] * rr[irow];
}

template <typename scalar, typename index>
void JacobiSRPreconditioner<scalar,index>::compute()
{
	if(!dblocks) {
		dblocks = new scalar[mat.nbrows];
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
	scalar_jacobi_apply(&mat, dblocks, rr, zz);
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
