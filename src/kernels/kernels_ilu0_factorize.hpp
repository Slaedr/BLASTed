/** \file
 * \brief Kernels for asynchronous scalar ILU factorization
 * \author Aditya Kashi
 */

#ifndef BLASTED_KERNELS_ILU0_FACTORIZE_H
#define BLASTED_KERNELS_ILU0_FACTORIZE_H

#include "../helper_algorithms.hpp"

namespace blasted {

/// Computes one row of an asynchronous ILU(0) factorization
/** Depending on template parameters, it can
 * factorize a scaled matrix, though the original matrix is not modified.
 * \note In the factorization loop, the variable pos is initially set negative.
 * If index is an unsigned type, that might be a problem. However,
 * it should usually be okay as we are only comparing equality later.
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
			//scalar sum = scale[irow] * mat->vals[j] * scale[mat->bcolind[j]];
			scalar sum = mat->vals[j];
			if(needscalerow)
				sum *= rowscale[irow];
			if(needscalecol)
				sum *= colscale[mat->bcolind[j]];

			for(index k = mat->browptr[irow]; 
				(k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
				k++  ) 
			{
				index pos = -1;
				internal::inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
						mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

				if(pos == -1) {
					continue;
				}

				sum -= iluvals[k]*iluvals[pos];
			}

			iluvals[j] = sum / iluvals[mat->diagind[mat->bcolind[j]]];
		}
		else
		{
			// compute u_ij
			//iluvals[j] = scale[irow]*mat->vals[j]*scale[mat->bcolind[j]];
			iluvals[j] = mat->vals[j];
			if(needscalerow)
				iluvals[j] *= rowscale[irow];
			if(needscalecol)
				iluvals[j] *= colscale[mat->bcolind[j]];

			for(index k = mat->browptr[irow]; 
					(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
			{
				index pos = -1;

				/* search for column index mat->bcolind[j], 
					* between the diagonal index of row mat->bcolind[k] 
					* and the last index of row mat->bcolind[k]
					*/
				internal::inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
										mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

				if(pos == -1) continue;

				iluvals[j] -= iluvals[k]*iluvals[pos];
			}
		}
	}
}

}

#endif

