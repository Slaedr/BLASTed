/** \file
 * \brief Asynchronous scalar ILU factorization implementation(s)
 * \author Aditya Kashi
 */

#include <iostream>
#include "async_ilu_factor.hpp"
#include "kernels/kernels_ilu0_factorize.hpp"
#include "helper_algorithms.hpp"

namespace blasted {

/// Initialize the factorization such that async. ILU(0) factorization gives async. SGS at worst
/** We set L' to (I+LD^(-1)) and U' to (D+U) so that L'U' = (D+L)D^(-1)(D+U).
 */
template <typename scalar ,typename index> static
void fact_init_sgs(const CRawBSRMatrix<scalar,index> *const mat, const scalar *const scale,
                   scalar *const __restrict iluvals)
{
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++)
	{
		for(index j = mat->browptr[i]; j < mat->browptr[i+1]; j++)
			iluvals[j] = scale[i]*mat->vals[j]*scale[mat->bcolind[j]];
		for(index j = mat->browptr[i]; j < mat->diagind[i]; j++)
			iluvals[j] *= 1.0/mat->vals[mat->diagind[mat->bcolind[j]]];
	}
}

/// Initialize the ILU factorization with the row-scaled original matrix
template <typename scalar ,typename index> static
void fact_init_scaled_original(const CRawBSRMatrix<scalar,index> *const mat,
                               const scalar *const scale,
                               scalar *const __restrict iluvals)
{
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++)
	{
		for(index j = mat->browptr[i]; j < mat->browptr[i+1]; j++)
			iluvals[j] = scale[i]*mat->vals[j]*scale[mat->bcolind[j]];
	}
}

template <typename scalar, typename index, bool scalerow, bool scalecol>
static void executeILU0Factorization(const CRawBSRMatrix<scalar,index> *const mat,
                                     const int nbuildsweeps, const int thread_chunk_size,
                                     const bool usethreads,
                                     const scalar *const rowscale, const scalar *const colscale,
                                     scalar *const __restrict iluvals)
{
#pragma omp parallel default(shared) if(usethreads)
	{
		for(int isweep = 0; isweep < nbuildsweeps; isweep++)
		{
#pragma omp for schedule(dynamic, thread_chunk_size)
			for(index irow = 0; irow < mat->nbrows; irow++)
			{
				async_ilu0_factorize_kernel<scalar,index,scalerow,scalecol>(mat, irow,
				                                                            rowscale, colscale,
				                                                            iluvals);
			}
		}
	}
}

template <typename scalar, typename index>
void scalar_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                           const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                           const FactInit factinittype,
                           scalar *const __restrict iluvals, scalar *const __restrict scale)
{
	// get the diagonal scaling matrix
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
		scale[i] = 1.0/std::sqrt(mat->vals[mat->diagind[i]]);

	switch(factinittype) {
	case INIT_F_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->browptr[mat->nbrows]; i++)
			iluvals[i] = 0;
	case INIT_F_ORIGINAL:
		fact_init_scaled_original(mat, scale, iluvals);
		break;
	case INIT_F_SGS:
		fact_init_sgs(mat, scale, iluvals);
		break;
	default:
		;
	}

	executeILU0Factorization<scalar,index,true,true>(mat, nbuildsweeps, thread_chunk_size,
	                                                 usethreads, scale, scale, iluvals);
}

template void
scalar_ilu0_factorize<double,int>(const CRawBSRMatrix<double,int> *const mat,
                                  const int nbuildsweeps, const int thread_chunk_size,
                                  const bool usethreads, const FactInit finit,
                                  double *const __restrict iluvals, double *const __restrict scale);

/////////////////////---//////////////////////

/// \todo TODO
template <typename scalar, typename index>
void scalar_ilu0_factorize_noscale(const CRawBSRMatrix<scalar,index> *const mat,
                                   const int nbuildsweeps, const int thread_chunk_size,
                                   const bool usethreads,
                                   scalar *const __restrict iluvals)
{
	/** initial guess
	 * We choose the initial guess such that the preconditioner reduces to SGS in the worst case.
	 */
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++)
	{
		for(index j = mat->browptr[i]; j < mat->browptr[i+1]; j++)
			iluvals[j] = mat->vals[j];
		/*for(index j = mat->browptr[i]; j < mat->diagind[i]; j++)
			iluvals[j] *= 1.0/mat->vals[mat->diagind[mat->bcolind[j]]];
		*/
	}

	// compute L and U
	executeILU0Factorization<scalar,index,false,false>(mat, nbuildsweeps, thread_chunk_size,
	                                                   usethreads, nullptr, nullptr, iluvals);
}

template
void scalar_ilu0_factorize_noscale<double,int>(const CRawBSRMatrix<double,int> *const mat,
                                               const int nbuildsweeps, const int thread_chunk_size,
                                               const bool usethreads,
                                               double *const __restrict iluvals);

}
