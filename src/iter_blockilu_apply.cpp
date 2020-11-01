/** \file iter_blockilu_apply.cpp
 * \brief Implementation of iterative triangular solvers for ILU factorizations
 * \author Aditya Kashi
 */

#include "iter_blockilu_apply.hpp"

#include <boost/align/aligned_alloc.hpp>
#include "kernels/kernels_ilu_apply.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
void async_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const Block_t<scalar,bs,stor> *const ilu,
                            const Segment_t<scalar,bs> *const z,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            Segment_t<scalar,bs> *const y);

template <typename scalar, typename index, int bs, StorageOptions stor>
void jacobi_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const Block_t<scalar,bs,stor> *const ilu,
                             const Segment_t<scalar,bs> *const z,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             Segment_t<scalar,bs> *const y);

template <typename scalar, typename index, int bs, StorageOptions stor>
void async_unitupper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const Block_t<scalar,bs,stor> *const ilu,
                            const Segment_t<scalar,bs> *const y,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            Segment_t<scalar,bs> *const z);

template <typename scalar, typename index, int bs, StorageOptions stor>
void jacobi_unitupper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const Block_t<scalar,bs,stor> *const ilu,
                             const Segment_t<scalar,bs> *const y,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             Segment_t<scalar,bs> *const z);

/// Applies the block-ILU0 factorization using a block variant of the asynch triangular solve in
/// \cite async:anzt_triangular
/**
 * \param[in] mat The BSR matrix
 * \param[in] iluvals The ILU factorization non-zeros, accessed using the block-row pointers,
 *   block-column indices and diagonal pointers of the original BSR matrix
 * \param ytemp A pre-allocated temporary vector, needed for applying the ILU0 factors
 * \param[in] napplysweeps Number of asynchronous sweeps to use for parallel application
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[in] usethreads Whether to use asynchronous threaded (true) or serial (false) application
 * \param[in] init_type Type of initialization
 * \param[in] jacobiiter Set to true to use Jacobi iterations for triangular solve instead of async
 * \param[in] r The RHS vector of the preconditioning problem Mz = r
 * \param[in,out] z The solution vector of the preconditioning problem Mz = r
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void block_ilu0_apply(const CRawBSRMatrix<scalar,index> *const mat,
                      const scalar *const iluvals, const scalar *const scale,
                      scalar *const __restrict y_temp,
                      const int napplysweeps, const int thread_chunk_size, const bool usethreads,
                      const ApplyInit init_type, const bool jacobiiter,
                      const scalar *const rr, scalar *const __restrict zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	const Blk *ilu = reinterpret_cast<const Blk*>(iluvals);
	//const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(y_temp);

	if(scale)
		// initially, z := Sr
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++) {
			zz[i] = scale[i]*rr[i];
		}
	else
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++) {
			zz[i] = rr[i];
		}

	switch(init_type) {
	case INIT_A_JACOBI:
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			y_temp[i] = 0;
		}
		break;
	default:
		;
	}

	// Do the lower sweeps
	if(jacobiiter)
		jacobi_unitlower_sweeps<scalar,index,bs,stor>(mat, ilu, z, usethreads, napplysweeps,
		                                              thread_chunk_size, y);
	else
		async_unitlower_sweeps<scalar,index,bs,stor>(mat, ilu, z, usethreads, napplysweeps,
		                                             thread_chunk_size, y);

	switch(init_type) {
	case INIT_A_JACOBI:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			zz[i] = y_temp[i];
		}
		break;
	case INIT_A_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->nbrows*bs; i++)
		{
			zz[i] = 0;
		}
		break;
	default:
		throw std::runtime_error(" scalar_ilu0_apply: Invalid init type!");
	}

	// Iterate upper
	if(jacobiiter)
		jacobi_unitupper_sweeps<scalar,index,bs,stor>(mat, ilu, y, usethreads, napplysweeps,
		                                              thread_chunk_size, z);
	else
		async_unitupper_sweeps<scalar,index,bs,stor>(mat, ilu, y, usethreads, napplysweeps,
		                                             thread_chunk_size, z);

	// scale z
	if(scale)
#pragma omp parallel for simd default(shared)
		for(int i = 0; i < mat->nbrows*bs; i++)
			zz[i] = zz[i]*scale[i];
}

template
void block_ilu0_apply<double,int,4,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                             const double *const iluvals, const double *const scale,
                                             double *const __restrict y_temp,
                                             const int napplysweeps, const int thread_chunk_size,
                                             const bool usethreads,
                                             const ApplyInit init_type, const bool jacobiiter,
                                             const double *const rr, double *const __restrict zz);
template
void block_ilu0_apply<double,int,4,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                             const double *const iluvals, const double *const scale,
                                             double *const __restrict y_temp,
                                             const int napplysweeps, const int thread_chunk_size,
                                             const bool usethreads,
                                             const ApplyInit init_type, const bool jacobiiter,
                                             const double *const rr, double *const __restrict zz);
template
void block_ilu0_apply<double,int,5,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                             const double *const iluvals, const double *const scale,
                                             double *const __restrict y_temp,
                                             const int napplysweeps, const int thread_chunk_size,
                                             const bool usethreads,
                                             const ApplyInit init_type, const bool jacobiiter,
                                             const double *const rr, double *const __restrict zz);


using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

/** solves Ly = Sr by asynchronous Jacobi iterations.
 * Note that if done serially, this is a forward-substitution.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void async_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const Block_t<scalar,bs,stor> *const ilu,
                            const Segment_t<scalar,bs> *const z,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            Segment_t<scalar,bs> *const y)
{
#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index i = 0; i < mat->nbrows; i++)
		{
			block_unit_lower_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->browptr[i], mat->diagind[i], z[i], i, y);
		}
	}
}

/** solves Ly = Sr by Jacobi iterations.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void jacobi_unitlower_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const Block_t<scalar,bs,stor> *const ilu,
                             const Segment_t<scalar,bs> *const z,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             Segment_t<scalar,bs> *const y)
{
	scalar *const yoldarr = static_cast<scalar*>(aligned_alloc(CACHE_LINE_LEN, mat->nbrows*bs*sizeof(scalar)));
	using Seg = Segment_t<scalar,bs>;
	Seg *const yold = reinterpret_cast<Seg*>(yoldarr);

#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for
		for(index i = 0; i < mat->nbrows*bs; i++)
			yold[i] = y[i];

#pragma omp for schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < mat->nbrows; i++)
		{
			jacobi_block_unit_lower_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->browptr[i], mat->diagind[i], z[i], i, yold, y);
		}
	}

	aligned_free(yoldarr);
}

/** Solves Uz = y by asynchronous Jacobi iteration.
 * If done serially, this is a back-substitution.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void async_unitupper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                            const Block_t<scalar,bs,stor> *const ilu,
                            const Segment_t<scalar,bs> *const y,
                            const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                            Segment_t<scalar,bs> *const z)
{
#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			block_upper_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->diagind[i], mat->browptr[i+1], y[i], i, z);
		}
	}
}

/** Solves Uz = y by Jacobi iteration.
 * 
 * Note that we now invert the loop direction for a possible small gain in efficiency.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void jacobi_unitupper_sweeps(const CRawBSRMatrix<scalar,index> *const mat,
                             const Block_t<scalar,bs,stor> *const ilu,
                             const Segment_t<scalar,bs> *const y,
                             const bool usethreads, const int napplysweeps, const int thread_chunk_size,
                             Segment_t<scalar,bs> *const z)
{
	scalar *const zoldarr = static_cast<scalar*>(aligned_alloc(CACHE_LINE_LEN, mat->nbrows*bs*sizeof(scalar)));
	using Seg = Segment_t<scalar,bs>;
	Seg *const zold = reinterpret_cast<Seg*>(zoldarr);

#pragma omp parallel default(shared) if(usethreads)
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp for
		for(index i = 0; i < mat->nbrows*bs; i++)
			zold[i] = z[i];

#pragma omp for schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < mat->nbrows; i++)
		{
			jacobi_block_upper_triangular<scalar,index,bs,stor>
				(ilu, mat->bcolind, mat->diagind[i], mat->browptr[i+1], y[i], i, zold, z);
		}
	}

	aligned_free(zoldarr);
}

}
