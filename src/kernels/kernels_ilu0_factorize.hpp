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
		if(irow > mat->bcolind[j])
		{
			scalar sum = mat->vals[j];
			if(needscalerow)
				sum *= rowscale[irow];
			if(needscalecol)
				sum *= colscale[mat->bcolind[j]];

			for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
			{
				sum -= iluvals[plist.lowerp[k]]*iluvals[plist.upperp[k]];
			}

			sum = sum / iluvals[mat->diagind[mat->bcolind[j]]];
			iluvals[j] = sum;
		}
		else
		{
			scalar sum = mat->vals[j];
			if(needscalerow)
				sum *= rowscale[irow];
			if(needscalecol)
				sum *= colscale[mat->bcolind[j]];

			/* Caution! Do not directly modify the shared variable iluvals[j] inside the loop below.
			 * It appears that we should write to it as few times as possible.
			 * Even using atomic updates, directly updating iluvals[j] every time leads to the exact
			 * solution not being a fixed point.
			 */
			for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
			{
				sum -= iluvals[plist.lowerp[k]]*iluvals[plist.upperp[k]];
			}

			iluvals[j] = sum;
		}
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

		if(irow > mat->bcolind[j])
		{
			// compute L_ij
			for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
				sum.noalias() -= ilu[plist.lowerp[k]]*ilu[plist.upperp[k]];

			ilu[j].noalias() = sum * ilu[mat->diagind[mat->bcolind[j]]].inverse();
		}
		else
		{
			// compute U_ij
			for(index k = plist.posptr[j]; k < plist.posptr[j+1]; k++)
				sum -= ilu[plist.lowerp[k]]*ilu[plist.upperp[k]];

			ilu[j].noalias() = sum;
		}
	}
}

// const Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)> *const mvals

}

#endif

