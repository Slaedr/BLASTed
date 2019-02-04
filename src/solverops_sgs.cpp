/** \file solverops_sgs.cpp
 * \brief Implementation for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#include <type_traits>
#include <boost/align/aligned_alloc.hpp>
#include "solverops_sgs.hpp"
#include "kernels/kernels_sgs.hpp"
#include "kernels/kernels_relaxation.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::
AsyncBlockSGS_SRPreconditioner(const int naswps, const ApplyInit apply_inittype,
                               const int threadchunksize)
	: ytemp{nullptr}, napplysweeps{naswps}, ainit{apply_inittype}, thread_chunk_size{threadchunksize}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::~AsyncBlockSGS_SRPreconditioner()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::compute()
{
	BJacobiSRPreconditioner<scalar,index,bs,stor>::compute();
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*bs*sizeof(scalar));

#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			ytemp[i] = 0;
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::apply(const scalar *const rr,
                                                                 scalar *const __restrict zz) const
{
	//const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(ytemp);

	if(ainit == INIT_A_JACOBI || ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			ytemp[i] = 0;
	
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)
		perform_block_fgs<scalar,index,bs,stor>(mat, dblks, thread_chunk_size, r, y);
	}

	if(ainit == INIT_A_JACOBI)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			zz[i] = ytemp[i];
	else if(ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows*bs; i++)
			zz[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)
		perform_block_bgs<scalar,index,bs,stor>(mat, dblks, thread_chunk_size, y, z);
	}
}

template<typename scalar, typename index, int bs, StorageOptions stor>
void AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>::apply_relax(const scalar *const bb, 
                                                                       scalar *const __restrict xx) const
{
	const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *b = reinterpret_cast<const Seg*>(bb);
	// the solution vector is wrapped in both a pointer to const segment and one to mutable segment
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *xmut = reinterpret_cast<Seg*>(xx);

#pragma omp parallel default(shared)
	{
	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			block_relax_kernel<scalar,index,bs,stor>
				(mvals, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks[irow], b[irow], x, x, xmut[irow]);
		}
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			block_relax_kernel<scalar,index,bs,stor>
				(mvals, mat.bcolind, irow, mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblks[irow], b[irow], x, x, xmut[irow]);
		}
	}
	}
}

template <typename scalar, typename index>
AsyncSGS_SRPreconditioner<scalar,index>::AsyncSGS_SRPreconditioner(const int naswps,
                                                                   const ApplyInit apply_inittype,
                                                                   const int threadchunksize)
	: ytemp{nullptr}, napplysweeps{naswps}, ainit{apply_inittype}, thread_chunk_size{threadchunksize}
{ }

template <typename scalar, typename index>
AsyncSGS_SRPreconditioner<scalar,index>::~AsyncSGS_SRPreconditioner()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::compute()
{
	JacobiSRPreconditioner<scalar,index>::compute();
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(scalar));

#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			ytemp[i] = 0;
	}
}

template <typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::apply(const scalar *const rr,
                                                    scalar *const __restrict zz) const
{
	if(ainit == INIT_A_JACOBI || ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			ytemp[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)
		perform_scalar_fgs(mat, dblocks, thread_chunk_size, rr, ytemp);
	}

	if(ainit == INIT_A_JACOBI)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			zz[i] = ytemp[i];
	else if(ainit == INIT_A_ZERO)
#pragma omp parallel for simd default(shared)
		for(index i = 0; i < mat.nbrows; i++)
			zz[i] = 0;

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)
		perform_scalar_bgs(mat, dblocks, thread_chunk_size, ytemp, zz);
	}
}

template<typename scalar, typename index>
void AsyncSGS_SRPreconditioner<scalar,index>::apply_relax(const scalar *const b, 
                                                          scalar *const __restrict x) const
{
#pragma omp parallel default(shared)
	for(int step = 0; step < solveparams.maxits; step++)
	{
#pragma omp for schedule(dynamic, thread_chunk_size) nowait		
		for(index irow = 0; irow < mat.nbrows; irow++)
		{
			x[irow] = scalar_relax<scalar,index>
				(mat.vals, mat.bcolind,
				 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblocks[irow], b[irow], x, x);
		}
#pragma omp for schedule(dynamic, thread_chunk_size) nowait		
		for(index irow = mat.nbrows-1; irow >= 0; irow--)
		{
			x[irow] = scalar_relax<scalar,index>
				(mat.vals, mat.bcolind, 
				 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
				 dblocks[irow], b[irow], x, x);
		}		
	}
}

template <typename scalar, typename index>
CSC_BGS_Preconditioner<scalar,index>::CSC_BGS_Preconditioner(const int naswps,
                                                             const int threadchunksize)
	: napplysweeps{naswps}, thread_chunk_size{threadchunksize},
	  cmat{nullptr, nullptr, nullptr, nullptr,0}
{ }

template <typename scalar, typename index>
CSC_BGS_Preconditioner<scalar,index>::~CSC_BGS_Preconditioner()
{
	destroyRawBSCMatrix(cmat);
}

template <typename scalar, typename index>
void CSC_BGS_Preconditioner<scalar,index>::compute()
{
	JacobiSRPreconditioner<scalar,index>::compute();

	destroyRawBSCMatrix(cmat);
	convert_BSR_to_BSC<scalar,index,1>(&mat, &cmat);
}

template <typename scalar, typename index>
void CSC_BGS_Preconditioner<scalar,index>::apply(const scalar *const rr,
                                                 scalar *const __restrict zz) const
{
	scalar *temp = (scalar*)aligned_alloc(CACHE_LINE_LEN, cmat.nbcols*sizeof(scalar));
#pragma omp parallel for simd
	for(index i = 0; i < cmat.nbcols; i++) {
		temp[i] = 0;
		//zz[i] = rr[i]*dblocks[i];
	}

#pragma omp parallel default(shared) // firstprivate(napplysweeps)
	{
		for(int isweep = 0; isweep < napplysweeps; isweep++)
		{
#pragma omp for simd
			for(index j = cmat.nbcols-1; j >= 0; j--) {
				temp[j] = rr[j]*dblocks[j];
			}

#pragma omp for schedule(dynamic, thread_chunk_size) nowait
			for(index j = cmat.nbcols-1; j >= 0; j--)
			{
				zz[j] = temp[j];
				for(index ii = cmat.diagind[j]-1; ii >= cmat.bcolptr[j]; ii--)
				{
					const index irow = cmat.browind[ii];
					const scalar term = cmat.vals[ii]*zz[j]*dblocks[irow];
#pragma omp atomic update
					temp[irow] -= term;
				}
			}
		}
	}

	aligned_free(temp);
}

template <typename scalar, typename index>
void CSC_BGS_Preconditioner<scalar,index>::apply_relax(const scalar *const rr,
                                                       scalar *const __restrict zz) const
{
	throw std::runtime_error("CSC_BGS relaxation not implemented!");
}

// instantiations

template class AsyncSGS_SRPreconditioner<double,int>;
template class CSC_BGS_Preconditioner<double,int>;

template class AsyncBlockSGS_SRPreconditioner<double,int,4,ColMajor>;
template class AsyncBlockSGS_SRPreconditioner<double,int,5,ColMajor>;

template class AsyncBlockSGS_SRPreconditioner<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class AsyncBlockSGS_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class AsyncBlockSGS_SRPreconditioner<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

} // end namespace
