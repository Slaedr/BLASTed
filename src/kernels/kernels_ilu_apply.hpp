/** \file kernels_ilu_apply.hpp
 * \brief Kernels for ILU application operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_ILU_APPLY_H
#define BLASTED_KERNELS_ILU_APPLY_H

#include "srmatrixdefs.hpp"

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
/**
 * \param vals The LU factorization matrix to apply - the diagonal blocks are assumed pre-inverted
 * \param bcolind Array of block-column indices
 * \param browstart Index into bcolind corresponding to the start of the row irow (see below)
 * \param bdiagind Index into bcolind that corresponds to the diagonal entry of row irow (see below)
 * \param rhs The block of the RHS vector corresponding to block-row irow
 * \param irow The block-component of the vector of unknowns x to be updated
 * \param x The output vector, whose irow'th block will be updated
 */
template <typename scalar, typename index, int bs, StorageOptions stor> inline
void block_unit_lower_triangular(const Block_t<scalar,bs,stor> *const vals,
                                 const index *const bcolind,
                                 const index browstart, const index bdiagind,
                                 const Segment_t<scalar,bs>& rhs,
                                 const index irow, Segment_t<scalar,bs> *const x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		inter += vals[jj]*x[bcolind[jj]];

	x[irow] = rhs - inter;
}

/// Upper triangular solve kernel
/**
 * \param vals The LU factorization matrix to apply - the diagonal blocks are assumed pre-inverted
 * \param bcolind Array of block-column indices
 * \param bdiagind Index into bcolind that corresponds to the diagonal entry of row irow (see below)
 * \param nextbrowstart Index into bcolind corresponding to the start of the next row after irow
 * \param rhs The block of the RHS vector corresponding to block-row irow
 * \param irow The block-component of the vector of unknowns x to be updated
 * \param x The output vector, whose irow'th block will be updated
 */
template <typename scalar, typename index, int bs, StorageOptions stor> inline
void block_upper_triangular(const Block_t<scalar,bs,stor> *const vals,
                            const index *const bcolind,
                            const index bdiagind, const int nextbrowstart,
                            const Segment_t<scalar,bs>& rhs,
                            const int irow, Segment_t<scalar,bs> *const x)
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
	
	// compute U z
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		inter += vals[jj] * x[bcolind[jj]];

	// compute z = D^(-1) (y - U z) for the irow-th block-segment of z
	x[irow] = vals[bdiagind] * ( rhs - inter );
}

}

#endif
