/** \file kernels_sgs.hpp
 * \brief Kernels for SGS preconditioning operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_SGS_H
#define BLASTED_KERNELS_SGS_H

#include "kernels_base.hpp"

namespace blasted {
	
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
		const index irow, const index browstart, const index bdiagind,
		Map<const Mattype>& diaginv, Map<const Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0)*x.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs)*x.SEG<bs>(bcolind[jj]*bs);

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
		const index irow, const index bdiagind, const int nextbrowstart,
		Map<const Mattype>& diaginv, Map<Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
	
	// compute U z
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs) * x.SEG<bs>(bcolind[jj]*bs);

	// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
	if(Mattype::IsRowMajor)
		x.SEG<bs>(irow*bs) = diaginv.BLK<bs,bs>(irow*bs,0) 
			* ( vals.BLK<bs,bs>(bdiagind*bs,0)*rhs.SEG<bs>(irow*bs) - inter );
	else
		x.SEG<bs>(irow*bs) = diaginv.BLK<bs,bs>(0,irow*bs) 
			* ( vals.BLK<bs,bs>(0,bdiagind*bs)*rhs.SEG<bs>(irow*bs) - inter );
}

}

#endif
