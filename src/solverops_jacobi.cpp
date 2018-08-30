/** \file solverops_jacobi.cpp
 * \brief Implementation for (block-) Jacobi operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include <iostream>
#include <boost/align/aligned_alloc.hpp>
#include <Eigen/LU>
#include "solverops_jacobi.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>::BJacobiSRPreconditioner()
	: dblocks{nullptr}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
BJacobiSRPreconditioner<scalar,index,bs,stor>::~BJacobiSRPreconditioner()
{
	// delete [] dblocks;
	//Eigen::aligned_allocator<scalar> aa;
	//aa.deallocate(dblocks, mat.nbrows*bs*bs);
	aligned_free(dblocks);
}
	
template <typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiSRPreconditioner<scalar,index,bs,stor>::compute()
{
	if(!dblocks) {
		// dblocks = new scalar[mat.nbrows*bs*bs];
		// Eigen::aligned_allocator<scalar> aa;
		// dblocks = aa.allocate(mat.nbrows*bs*bs);
		dblocks = (scalar*)aligned_alloc(CACHE_LINE_ALIGNMENT,
		                                 mat.nbrows*bs*bs*sizeof(scalar));
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
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
		zz[irow] = dblocks[irow] * rr[irow];
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
