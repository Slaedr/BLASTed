/** \file
 * \brief Kernels for asynchronous scalar ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_KERNELS_ILU0_FACTORIZE_H
#define BLASTED_KERNELS_ILU0_FACTORIZE_H

#include <Eigen/LU>
#include "ilu_pattern.hpp"

namespace blasted {

/// Computes one row of an asynchronous ILU(0) factorization
/** Depending on template parameters, it can
 * factorize a scaled matrix, though the original matrix is not modified.
 * \param[in] plist Lists of positions in the LU matrix required for the ILU computation
 */
template <typename scalar, typename index, bool needscalerow, bool needscalecol> inline
void async_ilu0_factorize_kernel(const CRawBSRMatrix<scalar,index> *const mat,
                                 const ILUPositions<index>& plist,
                                 const index irow,
                                 const scalar *const rowscale, const scalar *const colscale,
                                 scalar *const __restrict iluvals)
{
	for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
	{
		scalar sum = mat->vals[j];
		if(needscalerow)
			sum *= rowscale[irow];
		if(needscalecol)
			sum *= colscale[mat->bcolind[j]];

		/* Caution! Do not directly modify the shared variable iluvals[j] inside the loop below.
		 * Even using atomic updates, directly updating iluvals[j] every time leads to an inconsistent
		 * fixed point iteration.
		 * It looks like iluvals[j] should never hold a temporary value, like a partial dot product.
		 * If another core reads iluvals[j] while it contains a temporary value, it would violate the
		 * Chazan-Miranker and Frommer-Szyld concepts of chaotic/asynchronous iteration.
		 */
		for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
		{
			sum -= iluvals[plist.lowerp[k]]*iluvals[plist.upperp[k]];
		}

		// For lower triangular part, divide by u_jj
		if(irow > mat->bcolind[j])
			sum = sum / iluvals[mat->diagind[mat->bcolind[j]]];

		iluvals[j] = sum;
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
inline void async_block_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                                       const Block_t<scalar,bs,stor> *const mvals,
                                       const ILUPositions<index>& plist, const index irow,
                                       Block_t<scalar,bs,stor> *const __restrict ilu)
{
	for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
	{
		Matrix<scalar,bs,bs> sum = mvals[j];

		for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
			sum.noalias() -= ilu[plist.lowerp[k]]*ilu[plist.upperp[k]];

		if(irow > mat->bcolind[j])
		{
			ilu[j].noalias() = sum * ilu[mat->diagind[mat->bcolind[j]]].inverse();
		}
		else
		{
			ilu[j].noalias() = sum;
		}
	}
}

}

#endif

