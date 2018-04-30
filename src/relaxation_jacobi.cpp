/** \file relaxation_base.cpp
 * \brief Implementation of Jacobi relaxation
 * \author Aditya Kashi
 * \date 2018-04
 */

#include "relaxation_jacobi.hpp"
#include "kernels/kernels_relaxation.hpp"

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
BJacobiRelaxation<scalar,index,bs,stor>::BJacobiRelaxation(const scalar reltol, const scalar abstol, 
		const scalar divtol, const bool checktol, const int maxiter)
	: rtol{reltol}, atol{abstol}, dtol{divtol}, ctol{checktol}, maxits{maxiter}
{ }

template<typename scalar, typename index, int bs, class Mattype>
void bjacobi_relax(const scalar rtol, const scalar atol, const scalar dtol, 
		const bool ctol, const int maxits,
		const CRawBSRMatrix<scalar,index>& mat, const scalar *const dblocks,
		const scalar *const bb, scalar *const __restrict xx)
{
	Vector<scalar> xtemp(mat.nbrows*bs);
	scalar refdiffnorm = 1;
	
	Eigen::Map<const Vector<scalar>> b(bb, mat.nbrows*bs);
	//Eigen::Map<Vector<scalar>> x(xx, mat.nbrows*bs);
	Eigen::Map<const Vector<scalar>> x(xx, mat.nbrows*bs);

	Eigen::Map<const Mattype> data(mat.vals, 
			Mattype::IsRowMajor ? mat.browptr[mat.nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat.browptr[mat.nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat.nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat.nbrows*bs
		);

	for(int step = 0; step < maxits; step++)
	{
#pragma omp parallel for default(shared)
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_relax<scalar,index,bs,Mattype,Vector<scalar>>(data, mat.bcolind, 
				irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				dblks, b, x, x, xtemp);
		}

		if(ctol)
		{
			scalar diffnorm = 0;
#pragma omp parallel for simd default(shared) reduction(+:diffnorm)
			for(index i = 0; i < mat.nbrows*bs; i++) 
			{
				scalar diff = xtemp.data()[i] - xx[i];
				diffnorm += diff*diff;
				xx[i] = xtemp.data()[i];
			}
			diffnorm = std::sqrt(diffnorm);

			if(step == 0)
				refdiffnorm = diffnorm;

			if(diffnorm < atol || diffnorm/refdiffnorm < rtol || diffnorm/refdiffnorm > dtol)
				break;
		}
		else
		{
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows*bs; i++) {
				xx[i] = xtemp.data()[i];
			}
		}
	}
}

template<typename scalar, typename index, int bs, StorageOptions stor>
void BJacobiRelaxation<scalar,index,bs,stor>::apply(const scalar *const b, 
		scalar *const __restrict x) const
{
	if(stor == RowMajor)
		bjacobi_relax<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
			( rtol, atol, dtol, ctol, maxits, mat, dblocks, b, x);
	else
		bjacobi_relax<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>
			( rtol, atol, dtol, ctol, maxits, mat, dblocks, b, x);
}

template class BJacobiRelaxation<double,int,4,RowMajor>;
template class BJacobiRelaxation<double,int,4,ColMajor>;

template class BJacobiRelaxation<double,int,5,ColMajor>;

#ifdef BUILD_BLOCK_SIZE
template class BJacobiRelaxation<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class BJacobiRelaxation<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

}
