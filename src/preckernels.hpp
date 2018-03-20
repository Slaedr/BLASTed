/** \file preckernels.hpp
 * \brief Kernels for some preconditioning operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_PRECKERNELS_H
#define BLASTED_PRECKERNELS_H

#include <Eigen/LU>

/// Shorthand for dependent templates for Eigen segment function for vectors
#define SEG template segment
/// Shorthand for dependent templates for Eigen block function for matrices
#define BLK template block

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Map;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// Forward Gauss-Seidel kernel
template <typename scalar, typename index> inline 
scalar scalar_fgs(const scalar *const __restrict vals, const index *const __restrict colind, 
		const index rowstart, const index diagind,
		const scalar diag_entry_inv, const scalar rhs, 
		const scalar *const __restrict x)
{
	scalar inter = 0;
	for(index jj = rowstart; jj < diagind; jj++)
		inter += vals[jj]*x[colind[jj]];

	return diag_entry_inv * (rhs - inter);
}

/// Backward Gauss-Seidel kernel
template <typename scalar, typename index> inline 
scalar scalar_bgs(const scalar *const __restrict vals, const index *const __restrict colind, 
		const index diagind, const index nextrowstart,
		const scalar diag_entry, const scalar diag_entry_inv, const scalar rhs, 
		const scalar *const __restrict x)
{
	scalar inter = 0;
	for(index jj = diagind+1; jj < nextrowstart; jj++)
		inter += vals[jj]*x[colind[jj]];

	return diag_entry_inv * (diag_entry*rhs - inter);
}

/// Forward block Gauss-Seidel kernel
template <typename scalar, typename index, int bs, class Mattype> inline
void block_fgs(Map<const Mattype>& vals, const index *const __restrict bcolind,
		const index browstart, const index bdiagind,
		Map<const Mattype>& diaginv, Map<Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0)*x.SEG<bs>(mat->bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs)*x.SEG<bs>(mat->bcolind[jj]*bs);

	if(Mattype::IsRowMajor)
		x.SEG<bs>(irow*bs) = diaginv.BLK<bs,bs>(irow*bs,0) 
											  * (rhs.SEG<bs>(irow*bs) - inter);
	else
		x.SEG<bs>(irow*bs) = diaginv.BLK<bs,bs>(0,irow*bs) 
											  * (rhs.SEG<bs>(irow*bs) - inter);
}

/// Backward block Gauss-Seidel kernel
template <typename scalar, typename index, int bs, class Mattype> inline
void block_bgs(Map<const Mattype>& vals, const index *const __restrict bcolind,
		const index bdiagind, const int nextbrowstart,
		Map<const Mattype>& diaginv, Map<Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
	
	// compute U z
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0) * z.SEG<bs>(mat->bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs) * z.SEG<bs>(mat->bcolind[jj]*bs);

	// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
	if(Mattype::IsRowMajor)
		z.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(irow*bs,0) 
			* ( vals.BLK<bs,bs>(bdiagind*bs,0)*rhs.SEG<bs>(irow*bs) - inter );
	else
		z.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(0,irow*bs) 
			* ( vals.BLK<bs,bs>(0,bdiagind*bs)*rhs.SEG<bs>(irow*bs) - inter );
}

}

#endif
