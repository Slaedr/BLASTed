/** \file iter_scalarilu_apply.cpp
 * \brief Iterative triangular solves for applying ILU factorizations
 */

#include "iter_scalarilu_apply.hpp"
#include "kernels/kernels_ilu_apply.hpp"

namespace blasted {

template <typename scalar,typename index>
void async_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const scalar *const iluvals, const scalar *const za,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            scalar *const __restrict ytemp);

template <typename scalar,typename index>
void jacobi_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const scalar *const iluvals, const scalar *const za,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             scalar *const __restrict ytemp);

template <typename scalar,typename index>
void async_upper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                        const scalar *const iluvals, const scalar *const ytemp,
                        const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                        scalar *const __restrict za);

template <typename scalar,typename index>
void jacobi_upper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                         const scalar *const iluvals, const scalar *const ytemp,
                         const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                         scalar *const __restrict za);

template <typename scalar,typename index>
void scalar_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                       const scalar *const iluvals, const scalar *const scale,
                       scalar *const __restrict ytemp,
                       const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                       const ApplyInit init_type, const bool jacobiiter,
                       const scalar *const ra, scalar *const __restrict za) 
{
	// initially, z := Sr
	if(scale)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = scale[i]*ra[i];
		}
	else
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = ra[i];
		}

	switch(init_type) {
	case INIT_A_JACOBI:
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			ytemp[i] = 0;
		}
	break;
	default:
		;
	}

	if(jacobiiter)
		jacobi_unitlower_sweeps(mat, iluvals, za, usethreads, napplysweeps, thread_chunk_size, ytemp);
	else
		async_unitlower_sweeps(mat, iluvals, za, usethreads, napplysweeps, thread_chunk_size, ytemp);

	switch(init_type) {
	case INIT_A_JACOBI:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = ytemp[i];
		}
		break;
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows; i++) {
			za[i] = 0;
		}
		break;
	default:
		throw std::runtime_error(" scalar_ilu0_apply: Invalid init type!");
	}

	if(jacobiiter)
		jacobi_upper_sweeps(mat, iluvals, ytemp, usethreads, napplysweeps, thread_chunk_size, za);
	else
		async_upper_sweeps(mat, iluvals, ytemp, usethreads, napplysweeps, thread_chunk_size, za);

	if(scale)
		// scale z
#pragma omp parallel for simd default(shared)
		for(int i = 0; i < mat->nbrows; i++)
			za[i] = za[i]*scale[i];
}

template
void scalar_ilu0_apply<double,int>(const CRawBSRMatrix<double,int> *const mat,
                                   const double *const iluvals, const double *const scale,
                                   double *const __restrict ytemp,
                                   const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                                   const ApplyInit init_type, const bool jacobiiter,
                                   const double *const ra, double *const __restrict za);

/** solves Ly = Sr by chaotic relaxation
 * Note that if done serially, this is a forward-substitution.
 */
template <typename scalar, typename index>
void async_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const scalar *const iluvals, const scalar *const za,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            scalar *const __restrict ytemp)
{
#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index i = 0; i < mat->nbrows; i++)
		{
			ytemp[i] = scalar_unit_lower_triangular(iluvals, mat->bcolind, mat->browptr[i],
			                                        mat->diagind[i], za[i], ytemp);
		}
	}
}

/** Solves Ly = Sr by synchronous Jacobi iterations.
 * Note that if done serially, this is a forward-substitution.
 */
template <typename scalar, typename index>
void jacobi_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const scalar *const iluvals, const scalar *const za,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             scalar *const __restrict ytemp)
{
	scalar *yold = static_cast<scalar*>(aligned_alloc(CACHE_LINE_LEN, mat->nbrows*sizeof(scalar)));

#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for simd
		for(index i = 0; i < mat->nbrows; i++)
			yold[i] = ytemp[i];

#pragma omp for schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < mat->nbrows; i++)
		{
			ytemp[i] = scalar_unit_lower_triangular(iluvals, mat->bcolind, mat->browptr[i],
			                                        mat->diagind[i], za[i], yold);
		}
	}

	aligned_free(yold);
}

/** Solves Uz = y by asynchronous iteration.
 * If done serially, this is a back-substitution.
 */
template <typename scalar, typename index>
void async_upper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                        const scalar *const iluvals, const scalar *const ytemp,
                        const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                        scalar *const __restrict za)
{
#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			za[i] = scalar_upper_triangular<scalar,index>(iluvals, mat->bcolind, mat->diagind[i],
			                                              mat->browptr[i+1], 1.0/iluvals[mat->diagind[i]],
			                                              ytemp[i], za);
		}
	}
}

/** Solves Uz = y by Jacobi iteration.
 * If done serially, this is a back-substitution.
 */
template <typename scalar, typename index>
void jacobi_upper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                         const scalar *const iluvals, const scalar *const ytemp,
                         const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                         scalar *const __restrict za)
{
	scalar *zold = static_cast<scalar*>(aligned_alloc(CACHE_LINE_LEN, mat->nbrows*sizeof(scalar)));

#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for simd
		for(index i = 0; i < mat->nbrows; i++)
			zold[i] = za[i];

#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index i = 0; i < mat->nbrows; i--)
		{
			za[i] = scalar_upper_triangular<scalar,index>(iluvals, mat->bcolind, mat->diagind[i],
			                                              mat->browptr[i+1], 1.0/iluvals[mat->diagind[i]],
			                                              ytemp[i], zold);
		}
	}

	aligned_free(zold);
}

}
