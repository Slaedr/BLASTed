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

namespace blasted {

/// Initialize the factorization such that async. ILU(0) factorization gives async. SGS at worst
/**
 * We set L' to (I+LD^(-1)) and U' to (D+U) so that L'U' = (D+L)D^(-1)(D+U).
 */
template <typename scalar, typename index, int bs, StorageOptions stor> static
void fact_init_sgs(const CRawBSRMatrix<scalar,index> *const mat, scalar *const __restrict iluvals)
{
	using NABlk = Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)>;
	using Blk = Block_t<scalar,bs,stor>;
	
	const NABlk *mvals = reinterpret_cast<const NABlk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	Eigen::aligned_allocator<Blk> alloc;
	Blk *dblks = alloc.allocate(mat->nbrows);

	// Initialize ilu to original matrix
#pragma omp parallel for default (shared)
	for(index i = 0; i < mat->nbrows; i++) {
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

/** \todo Revisit the requirement for non-aligned block types for the original matrix
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void block_ilu0_factorize(const CRawBSRMatrix<scalar,index> *const mat,
                          const int nbuildsweeps, const int thread_chunk_size, const bool usethreads,
                          const FactInit init_type,
                          scalar *const __restrict iluvals)
{
	using NABlk = Block_t<scalar,bs,static_cast<StorageOptions>(stor|Eigen::DontAlign)>;
	using Blk = Block_t<scalar,bs,stor>;
	
	const NABlk *mvals = reinterpret_cast<const NABlk*>(mat->vals);
	Blk *ilu = reinterpret_cast<Blk*>(iluvals);

	switch(init_type)
	{
	case INIT_F_ZERO:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->browptr[mat->nbrows]*bs*bs; i++)
			iluvals[i] = 0;
		break;
	case INIT_F_ORIGINAL:
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat->browptr[mat->nbrows]*bs*bs; i++)
			iluvals[i] = mat->vals[i];
		break;
	case INIT_F_SGS:
		fact_init_sgs<scalar,index,bs,stor>(mat, iluvals);
		break;
	default:;
		// do nothing
	}

	// compute L and U
	/* Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
			{
				if(irow > mat->bcolind[j])
				{
					Matrix<scalar,bs,bs> sum = mvals[j];

					for( index k = mat->browptr[irow]; 
						 (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
						 k++
					   ) 
					{
						index pos = -1;
						internal::inner_search<index>(mat->bcolind,
						                              mat->diagind[mat->bcolind[k]],
						                              mat->browptr[mat->bcolind[k]+1],
						                              mat->bcolind[j], &pos );

						if(pos == -1) continue;

						sum.noalias() -= ilu[k]*ilu[pos];
					}

					ilu[j].noalias() = sum * ilu[mat->diagind[mat->bcolind[j]]].inverse();
				}
				else
				{
					// compute u_ij
					ilu[j] = mvals[j];

					for(index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index mat->bcolind[j],
						 * between the diagonal index of row mat->bcolind[k] 
						 * and the last index of row bcolind[k]
						 */
						internal::inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
						                       mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

						if(pos == -1) continue;

						ilu[j].noalias() -= ilu[k]*ilu[pos];
					}
				}
			}
		}
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		ilu[mat->diagind[irow]] = ilu[mat->diagind[irow]].inverse().eval();
}

template void
block_ilu0_factorize<double,int,4,ColMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             double *const __restrict iluvals);
template void
block_ilu0_factorize<double,int,5,ColMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             double *const __restrict iluvals);
template void
block_ilu0_factorize<double,int,4,RowMajor> (const CRawBSRMatrix<double,int> *const mat,
                                             const int nbuildsweeps, const int thread_chunk_size,
                                             const bool usethreads, const FactInit inittype,
                                             double *const __restrict iluvals);

#ifdef BUILD_BLOCK_SIZE
template void block_ilu0_factorize<double,int,BUILD_BLOCK_SIZE,ColMajor>
(const CRawBSRMatrix<double,int> *const mat,
 const int nbuildsweeps, const int thread_chunk_size, const bool usethreads, const FactInit inittype,
 double *const __restrict iluvals);
#endif

template <typename scalar, typename index, int bs, StorageOptions stor>
void compute_ILU_residual(const CRawBSRMatrix<scalar,index> *const mat, const scalar *const iluvals,
                          const int thread_chunk_size,
                          scalar *const __restrict resvals)
{
	using Blk = Block_t<scalar,bs,stor>;

	const Blk *const mvals = reinterpret_cast<const Blk*>(mat->vals);
	const Blk *const ilu = reinterpret_cast<const Blk*>(iluvals);
	Blk *const res = reinterpret_cast<Blk*>(resvals);

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
		{
			if(irow > mat->bcolind[j])
			{
				res[j] = mvals[j];

				for( index k = mat->browptr[irow]; 
				     (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
				     k++
				     ) 
				{
					index pos = -1;
					internal::inner_search<index>(mat->bcolind,
					                              mat->diagind[mat->bcolind[k]],
					                              mat->browptr[mat->bcolind[k]+1],
					                              mat->bcolind[j], &pos );

					if(pos == -1) continue;

					res[j].noalias() -= ilu[k]*ilu[pos];
				}
			}
			else
			{
				// compute u_ij
				res[j] = mvals[j];

				for(index k = mat->browptr[irow]; 
				    (k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
				{
					index pos = -1;

					/* search for column index mat->bcolind[j],
					 * between the diagonal index of row mat->bcolind[k] 
					 * and the last index of row bcolind[k]
					 */
					internal::inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
					                       mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

					if(pos == -1) continue;

					res[j].noalias() -= ilu[k]*ilu[pos];
				}
			}
		}
	}
}

template
void compute_ILU_residual<double,int,4,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                                 const double *const iluvals,
                                                 const int thread_chunk_size,
                                                 double *const __restrict resvals);
template
void compute_ILU_residual<double,int,4,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                                 const double *const iluvals,
                                                 const int thread_chunk_size,
                                                 double *const __restrict resvals);
template
void compute_ILU_residual<double,int,5,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                                 const double *const iluvals,
                                                 const int thread_chunk_size,
                                                 double *const __restrict resvals);

}
