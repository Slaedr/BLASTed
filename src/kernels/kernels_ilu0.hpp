/** \file kernels_ilu0.hpp
 * \brief Kernels for ILU0 preconditioning operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_ILU0_H
#define BLASTED_KERNELS_ILU0_H

#include "kernels_base.hpp"

namespace blasted {
	
/// Unit lower triangular solve
template <typename scalar, typename index> inline 
scalar scalar_unit_lower_triangular(const scalar *const __restrict vals, 
		const index *const __restrict colind, const index rowstart, const index diagind,
		const scalar rhs, const scalar *const __restrict x)
{
	scalar inter = 0;
	for(index jj = rowstart; jj < diagind; jj++)
		inter += vals[jj]*x[colind[jj]];

	return rhs - inter;
}

/// Upper triangular solve kernel
template <typename scalar, typename index> inline 
scalar scalar_upper_triangular(const scalar *const __restrict vals, 
		const index *const __restrict colind, const index diagind, const index nextrowstart,
		const scalar diag_entry_inv, const scalar rhs, 
		const scalar *const __restrict x)
{
	scalar inter = 0;
	for(index jj = diagind+1; jj < nextrowstart; jj++)
		inter += vals[jj]*x[colind[jj]];

	return diag_entry_inv * (rhs - inter);
}

/// Unit lower triangular solve kernel
template <typename scalar, typename index, int bs, class Mattype> inline
void block_unit_lower_triangular(Map<const Mattype>& vals, const index *const __restrict bcolind,
		const index irow, const index browstart, const index bdiagind,
		Map<const Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0)*x.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs)*x.SEG<bs>(bcolind[jj]*bs);

	x.SEG<bs>(irow*bs) = rhs.SEG<bs>(irow*bs) - inter;
}

/// Upper triangular solve kernel
/**
 * \param vals The LU factorization matrix to apply - the diagonal blocks are assumed pre-inverted
 */
template <typename scalar, typename index, int bs, class Mattype> inline
void block_upper_triangular(Map<const Mattype>& vals, const index *const __restrict bcolind,
		const index irow, const index bdiagind, const int nextbrowstart,
		Map<Vector<scalar>>& rhs,  Map<Vector<scalar>>& x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
	
	// compute U z
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs) * x.SEG<bs>(bcolind[jj]*bs);

	// compute z = D^(-1) (y - U z) for the irow-th block-segment of z
	if(Mattype::IsRowMajor)
		x.SEG<bs>(irow*bs) = vals.BLK<bs,bs>(bdiagind*bs,0) * ( rhs.SEG<bs>(irow*bs) - inter );
	else
		x.SEG<bs>(irow*bs) = vals.BLK<bs,bs>(0,bdiagind*bs) * ( rhs.SEG<bs>(irow*bs) - inter );
}

}

#endif
