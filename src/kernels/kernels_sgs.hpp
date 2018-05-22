/** \file kernels_sgs.hpp
 * \brief Kernels for SGS preconditioning operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_SGS_H
#define BLASTED_KERNELS_SGS_H

#include <Eigen/Core>

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

	return rhs - diag_entry_inv*inter;
}

/// Forward block Gauss-Seidel kernel
template <typename scalar, typename index, int bs, StorageOptions stor> inline
void block_fgs(const Block_t<scalar,bs,stor> *const vals, const index *const bcolind,
		const index irow, const index browstart, const index bdiagind,
		const Block_t<scalar,bs,stor>& diaginv, const Segment_t<scalar,bs>& rhs,
		Segment_t<scalar,bs> *const x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		inter += vals[jj]*x[bcolind[jj]];

	x[irow] = diaginv * (rhs - inter);
}

/// Backward block Gauss-Seidel kernel
template <typename scalar, typename index, int bs, StorageOptions stor> inline
void block_bgs(const Block_t<scalar,bs,stor> *const vals, const index *const bcolind,
		const index irow, const index bdiagind, const index nextbrowstart,
		const Block_t<scalar,bs,stor>& diaginv, const Segment_t<scalar,bs>& rhs,
		Segment_t<scalar,bs> *const x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
	
	// compute U z
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		inter += vals[jj] * x[bcolind[jj]];

	// compute z =  (y - D^(-1)*U z) for the irow-th block-segment of z
	x[irow] = rhs - diaginv*inter;
}

}

#endif
