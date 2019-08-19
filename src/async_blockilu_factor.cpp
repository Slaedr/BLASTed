/** \file
 * \brief Implementation(s) of asynchronous block-ILU factorization
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <Eigen/LU>
#include "async_blockilu_factor.hpp"
#include "helper_algorithms.hpp"
#include "kernels/kernels_ilu0_factorize.hpp"
#include "blas/blas1.hpp"
#include "matrix_properties.hpp"

#include <boost/align.hpp>
#include <iostream>

namespace blasted {

/// Initialize the factorization such that async. ILU(0) factorization gives async. SGS at worst
/**
 * We set L' to (I+LD^(-1)) and U' to (D+U) so that L'U' = (D+L)D^(-1)(D+U).
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
static void fact_init_sgs(const CRawBSRMatrix<scalar,index> *const mat, const scalar *const scale,
                          scalar *const __restrict iluvals);

/// Carry out the nonlinear asynchronous iterations to compute the ILU factors
template <typename scalar, typename index, int bs, StorageOptions stor, bool usescaling>
void async_bilu0_sweeps(const CRawBSRMatrix<scalar,index> *const mat, const ILUPositions<index>& plist,
                        const scalar *const scale, const int nbuildsweeps, const int thread_chunk_size,
                        const bool usethreads, scalar *const __restrict iluvals);

template <typename scalar, typename index, int bs, StorageOptions stor>
PrecInfo block_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                              const ILUPositions<index>& plist,
                              const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                              const FactInit init_type, const bool compute_info,
                              scalar *const __restrict iluvals, scalar *const __restrict scale)
{
	//using NABlk = Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)>;

	using Blk = Block_t<scalar,bs,stor>;
	const Blk *mvals = reinterpret_cast<const Blk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	// get the diagonal scaling matrix
	if(scale)
		getScalingVector<scalar,index,bs>(mat, scale);

	switch(init_type)
	{
	case INIT_F_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->browptr[mat->nbrows]*bs*bs; i++)
			iluvals[i] = 0;
		break;

	case INIT_F_ORIGINAL:
		if(scale)
#pragma omp parallel for default(shared)
			for(index irow = 0; irow < mat->nbrows; irow++)
			{
				for(index jj = mat->browptr[irow]; jj < mat->browendptr[irow]; jj++) {
					ilu[jj] = mvals[jj];
					scaleBlock<scalar,index,bs,stor>(scale, irow, mat->bcolind[jj], ilu[jj]);
				}
			}
		else
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat->browptr[mat->nbrows]*bs*bs; i++)
					iluvals[i] = mat->vals[i];
		break;

	case INIT_F_SGS:
		fact_init_sgs<scalar,index,bs,stor>(mat, scale, iluvals);
		break;

	default:;
		// do nothing
	}

	PrecInfo pinfo;

	if(compute_info)
	{
		if(scale)
			pinfo.prec_rem_initial_norm()
				= block_ilu0_nonlinear_res<scalar,index,bs,stor,true>(mat, plist, scale, iluvals,
				                                                      thread_chunk_size);
		else
			pinfo.prec_rem_initial_norm()
				= block_ilu0_nonlinear_res<scalar,index,bs,stor,false>(mat, plist, scale, iluvals,
				                                                       thread_chunk_size);
	}

	// compute L and U
	/* Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */

	if(scale)
		async_bilu0_sweeps<scalar,index,bs,stor,true>(mat, plist, scale, nbuildsweeps,
		                                              thread_chunk_size, usethreads, iluvals);
	else
		async_bilu0_sweeps<scalar,index,bs,stor,false>(mat, plist, scale, nbuildsweeps,
		                                               thread_chunk_size, usethreads, iluvals);

	if(compute_info)
	{
		if(scale)
			pinfo.prec_remainder_norm()
				= block_ilu0_nonlinear_res<scalar,index,bs,stor,true>(mat, plist, scale, iluvals,
				                                                      thread_chunk_size);
		else
			pinfo.prec_remainder_norm()
				= block_ilu0_nonlinear_res<scalar,index,bs,stor,false>(mat, plist, scale, iluvals,
				                                                       thread_chunk_size);

		std::array<scalar,2> arr = diagonal_dominance_lower<scalar,index,bs,stor>
			(SRMatrixStorage<const scalar,const index>(mat->browptr, mat->bcolind, iluvals,
			                                           mat->diagind, mat->browendptr, mat->nbrows,
			                                           mat->nnzb, mat->nbstored, bs));
		pinfo.lower_avg_diag_dom() = arr[0];
		pinfo.lower_min_diag_dom() = arr[1];

		std::array<scalar,2> uarr = diagonal_dominance_upper<scalar,index,bs,stor>
			(SRMatrixStorage<const scalar,const index>(mat->browptr, mat->bcolind, iluvals,
			                                           mat->diagind, mat->browendptr, mat->nbrows,
			                                           mat->nnzb, mat->nbstored, bs));

		pinfo.upper_avg_diag_dom() = uarr[0];
		pinfo.upper_min_diag_dom() = uarr[1];
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		ilu[mat->diagind[irow]] = ilu[mat->diagind[irow]].inverse().eval();

	return pinfo;
}

template PrecInfo
block_ilu0_factorize<double,int,4,ColMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const ILUPositions<int>& plist,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             const bool compute_residuals,
                                             double *const __restrict iluvals,
                                             double *const __restrict scale);
template PrecInfo
block_ilu0_factorize<double,int,5,ColMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const ILUPositions<int>& plist,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             const bool compute_residuals,
                                             double *const __restrict iluvals,
                                             double *const __restrict scale);
template PrecInfo
block_ilu0_factorize<double,int,4,RowMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const ILUPositions<int>& plist,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             const bool compute_residuals,
                                             double *const __restrict iluvals,
                                             double *const __restrict scale);

#ifdef BUILD_BLOCK_SIZE

template PrecInfo block_ilu0_factorize<double,int,BUILD_BLOCK_SIZE,ColMajor>

(const CRawBSRMatrix<double,int> *const mat, const ILUPositions<int>& plist,
 const int nbuildsweeps, const int thread_chunk_size, const bool usethreads, const FactInit inittype,
 const bool compute_residuals, double *const __restrict iluvals, double *const __restrict scale);

#endif

template <typename scalar, typename index, int bs, StorageOptions stor, bool usescaling>
void async_bilu0_sweeps(const CRawBSRMatrix<scalar,index> *const mat, const ILUPositions<index>& plist,
                        const scalar *const scale, const int nbuildsweeps, const int thread_chunk_size,
                        const bool usethreads,
                        scalar *const __restrict iluvals)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *mvals = reinterpret_cast<const Blk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
			async_block_ilu0_factorize<scalar,index,bs,stor,usescaling>(mat, mvals, plist, scale,
			                                                            irow, ilu);
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor> static
void fact_init_sgs(const CRawBSRMatrix<scalar,index> *const mat, const scalar *const scale,
                   scalar *const __restrict iluvals)
{
	//using NABlk = Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)>;
	using Blk = Block_t<scalar,bs,stor>;

	//const NABlk *mvals = reinterpret_cast<const NABlk*>(mat->vals);
	const Blk *mvals = reinterpret_cast<const Blk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	Eigen::aligned_allocator<Blk> alloc;
	Blk *dblks = alloc.allocate(mat->nbrows);

	// Initialize ilu to original matrix
	if(scale)
#pragma omp parallel for default (shared)
		for(index i = 0; i < mat->nbrows; i++)
		{
			Block_t<scalar,bs,stor> tempblk = mvals[mat->diagind[i]];
			scaleBlock<scalar,index,bs,stor>(scale, i, i, tempblk);
			dblks[i].noalias() = tempblk.inverse();

			for(index j = mat->browptr[i]; j < mat->browptr[i+1]; j++)
			{
				ilu[j] = mvals[j];
				scaleBlock<scalar,index,bs,stor>(scale, i, mat->bcolind[j], ilu[j]);
			}
		}
	else
#pragma omp parallel for default (shared)
		for(index i = 0; i < mat->nbrows; i++)
		{
			dblks[i].noalias() = mvals[mat->diagind[i]].inverse();

			for(index j = mat->browptr[i]; j < mat->browptr[i+1]; j++)
				ilu[j] = mvals[j];
		}

	// Scale the (strictly) lower block-triangular part
	//  by the inverses of the diagonal blocks from the right
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++) {
		for(index j = mat->browptr[i]; j < mat->diagind[i]; j++)
			ilu[j] = ilu[j]*dblks[mat->bcolind[j]].eval();
	}

	alloc.deallocate(dblks, mat->nbrows);
}

template <typename scalar, typename index, int bs, StorageOptions stor, bool usescaling>
scalar block_ilu0_nonlinear_res(const CRawBSRMatrix<scalar,index> *const mat,
                                const ILUPositions<index>& plist, const scalar *const scale,
                                const scalar *const iluvals,
                                const int thread_chunk_size)
{
	//using Blk = Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)>;
	using Blk = Block_t<scalar,bs,stor>;

	const Blk *const mvals = reinterpret_cast<const Blk*>(mat->vals);
	const Blk *const ilu = reinterpret_cast<const Blk*>(iluvals);

	scalar resnorm = 0;
	scalar anorm = 0;      // original matrix

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) reduction(+:resnorm,anorm)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			Block_t<scalar,bs,stor> sum = mvals[jj];
			if(usescaling)
				scaleBlock<scalar,index,bs,stor>(scale, irow, mat->bcolind[jj], sum);

			for(index k = plist.posptr[jj]; k < plist.posptr[jj+1]; k++)
				sum -= ilu[plist.lowerp[k]]*ilu[plist.upperp[k]];

			if(irow > mat->bcolind[jj])
				sum -= ilu[jj] * ilu[mat->diagind[mat->bcolind[jj]]];
			else
				sum -= ilu[jj];

			// Take the vector 1-norm of all non-zero entries
			scalar blockresnorm = 0;
			scalar blockanorm = 0;
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
				{
					blockresnorm += std::abs(sum(i,j));
					blockanorm += std::abs(mvals[jj](i,j));
				}

			resnorm += blockresnorm;
			anorm += blockanorm;
		}
	}

	return resnorm;
}

template
double block_ilu0_nonlinear_res<double,int,4,ColMajor,true>(const CRawBSRMatrix<double,int> *const mat,
                                                            const ILUPositions<int>& plist,
                                                            const double *const scale,
                                                            const double *const iluvals,
                                                            const int thread_chunk_size);
template
double block_ilu0_nonlinear_res<double,int,4,ColMajor,false>(const CRawBSRMatrix<double,int> *const mat,
                                                             const ILUPositions<int>& plist,
                                                             const double *const scale,
                                                             const double *const iluvals,
                                                             const int thread_chunk_size);

}
