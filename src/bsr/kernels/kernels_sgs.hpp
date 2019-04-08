/** \file kernels_sgs.hpp
 * \brief Kernels for SGS preconditioning operations
 * \author Aditya Kashi
 * \date 2018-03
 */

#ifndef BLASTED_KERNELS_SGS_H
#define BLASTED_KERNELS_SGS_H

#include "bsr/srmatrixdefs.hpp"

namespace blasted {

namespace kernels {

/// Forward Gauss-Seidel kernel (for one component of the solution vector)
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

/// Backward Gauss-Seidel kernel (for one component of the solution vector)
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

/// Forward block Gauss-Seidel kernel (for one block-component of the solution vector)
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

/// Backward block Gauss-Seidel kernel (for one block-component of the solution vector)
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

} // end kernels

/// Forward Gauss-Seidel solve
template <typename scalar, typename index>
void perform_scalar_fgs(const CRawBSRMatrix<scalar,index>& mat, const scalar *const diaginv,
                        const int thread_chunk_size,
                        const scalar *const rr, scalar *const __restrict ytemp)
{
	// forward sweep ytemp := D^(-1) (r - L ytemp)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		ytemp[irow] = kernels::scalar_fgs(mat.vals, mat.bcolind, mat.browptr[irow], mat.diagind[irow],
		                                  diaginv[irow], rr[irow], ytemp);
	}
}

/// Forward Gauss-Seidel solve
template <typename scalar, typename index>
void perform_scalar_bgs(const CRawBSRMatrix<scalar,index>& mat, const scalar *const diaginv,
                        const int thread_chunk_size,
                        const scalar *const ytemp, scalar *const __restrict zz)
{
	// backward sweep z := D^(-1) (D y - U z)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
	for(index irow = mat.nbrows-1; irow >= 0; irow--)
	{
		zz[irow] = kernels::scalar_bgs(mat.vals, mat.bcolind, mat.diagind[irow], mat.browptr[irow+1],
		                               mat.vals[mat.diagind[irow]], diaginv[irow], ytemp[irow], zz);
	}
}

/// Forward Gauss-Seidel solve
/** \param mat The original matrix
 * \param diagblks Inverses of all diagonal blocks of the original matrix
 * \param thread_chunk_size Number of work-items in each 'chunk' of work-items
 * \param r RHS vector
 * \param y Solution vector
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void perform_block_fgs(const CRawBSRMatrix<scalar,index>& mat,
                       const Block_t<scalar,bs,stor> *const diagblks,
                       const int thread_chunk_size,
                       const Segment_t<scalar,bs> *const r, Segment_t<scalar,bs> *const y)
{
	const Block_t<scalar,bs,stor> *const mvals
		= reinterpret_cast<const Block_t<scalar,bs,stor>*>(mat.vals);

	// forward sweep ytemp := D^(-1) (r - L ytemp)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		kernels::block_fgs<scalar,index,bs,stor>(mvals, mat.bcolind, irow, mat.browptr[irow], 
		                                         mat.diagind[irow], diagblks[irow], r[irow], y);
	}
}

/// Backward Gauss-Seidel solve
/** \param mat The original matrix
 * \param diagblks Inverses of all diagonal blocks of the original matrix
 * \param thread_chunk_size Number of work-items in each 'chunk' of work-items
 * \param y RHS vector
 * \param z Solution vector
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void perform_block_bgs(const CRawBSRMatrix<scalar,index>& mat,
                       const Block_t<scalar,bs,stor> *const diagblks,
                       const int thread_chunk_size,
                       const Segment_t<scalar,bs> *const y, Segment_t<scalar,bs> *const z)
{
	const Block_t<scalar,bs,stor> *const mvals
		= reinterpret_cast<const Block_t<scalar,bs,stor>*>(mat.vals);

	// backward sweep z := D^(-1) (D y - U z)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
	for(index irow = mat.nbrows-1; irow >= 0; irow--)
	{
		kernels::block_bgs<scalar, index, bs, stor>(mvals, mat.bcolind, irow, mat.diagind[irow],
		                                            mat.browptr[irow+1], diagblks[irow], y[irow], z);
	}
}

}

#endif
