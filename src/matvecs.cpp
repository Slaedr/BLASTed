/** \file
 * \brief Implementation of some basic kernels like spmv
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

#include "matvecs.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
void bsr_matrix_apply(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar *const xx, scalar *const __restrict yy)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	const Blk *data = reinterpret_cast<const Blk*>(mat->vals);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *y = reinterpret_cast<Seg*>(yy);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		y[irow] = Vector<scalar>::Zero(bs);

		// loop over non-zero blocks of this block-row
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = mat->bcolind[jj];
			y[irow].noalias() += data[jj] * x[jcol];
		}
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void bsr_gemv3(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	const Blk *data = reinterpret_cast<const Blk*>(mat->vals);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	const Seg *y = reinterpret_cast<const Seg*>(yy);
	Seg *z = reinterpret_cast<Seg*>(zz);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		z[irow] = b * y[irow];

		// loop over non-zero blocks of this block-row
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			const index jcol = mat->bcolind[jj];
			z[irow].noalias() += a * data[jj] * x[jcol];
		}
	}
}

template <typename scalar, typename index>
void csr_matrix_apply(const CRawBSRMatrix<scalar,index> *const mat,
                      const scalar *const xx, scalar *const __restrict yy) 
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		yy[irow] = 0;

		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			yy[irow] += mat->vals[jj] * xx[mat->bcolind[jj]];
		}
	}
}

template <typename scalar, typename index>
void csr_gemv3(const CRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz)
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		zz[irow] = b * yy[irow];

		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			zz[irow] += a * mat->vals[jj] * xx[mat->bcolind[jj]];
		}
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void bcsc_gemv3(const CRawBSCMatrix<scalar,index> *const mat,
                const scalar a, const scalar *const __restrict xx, 
                const scalar b, const scalar *const yy, scalar *const zz)
{
	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	const Blk *data = reinterpret_cast<const Blk*>(mat->vals);
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	//const Seg *y = reinterpret_cast<const Seg*>(yy);
	//Seg *z = reinterpret_cast<Seg*>(zz);

#pragma omp parallel default (shared)
	{
#pragma omp for simd
		for(index jb = 0; jb < mat->nbcols*bs; jb++)
			zz[jb] = b*yy[jb];

#pragma omp for
		for(index jcol = 0; jcol < mat->nbcols; jcol++)
		{
			// loop over non-zero blocks of this block-row
			for(index ii = mat->bcolptr[jcol]; ii < mat->bcolptr[jcol+1]; ii++)
			{
				const index irow = mat->browind[ii];
				const Matrix<scalar,bs,1> inter = a * data[ii] * x[jcol];
				//Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
				//inter = a * data[ii] * x[jcol];

				for(int kb = 0; kb < bs; kb++) {
#pragma omp atomic
					zz[irow*bs+kb] += inter[kb];
				}
			}
		}
	}
}

// Instantiations

template void
bsr_matrix_apply<double,int,3,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,4,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,7,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,3,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,4,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,5,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);
template void
bsr_matrix_apply<double,int,7,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                        const double *const xx, double *const __restrict yy);

template
void bsr_gemv3<double,int,3,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,4,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,7,RowMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,3,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,4,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,5,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);
template
void bsr_gemv3<double,int,7,ColMajor>(const CRawBSRMatrix<double,int> *const mat,
                                      const double a, const double *const __restrict xx,
                                      const double b, const double *const yy, double *const zz);

template
void csr_matrix_apply<double,int>(const CRawBSRMatrix<double,int> *const mat,
                                  const double *const xx, double *const __restrict yy);

template
void csr_gemv3<double,int>(const CRawBSRMatrix<double,int> *const mat,
                           const double a, const double *const __restrict xx, 
                           const double b, const double *const yy, double *const zz);

// BSC matrix

template
void bcsc_gemv3<double,int,7,RowMajor>(const CRawBSCMatrix<double,int> *const mat,
                                       const double a, const double *const __restrict xx,
                                       const double b, const double *const yy, double *const zz);
template
void bcsc_gemv3<double,int,3,ColMajor>(const CRawBSCMatrix<double,int> *const mat,
                                       const double a, const double *const __restrict xx,
                                       const double b, const double *const yy, double *const zz);
template
void bcsc_gemv3<double,int,4,ColMajor>(const CRawBSCMatrix<double,int> *const mat,
                                       const double a, const double *const __restrict xx,
                                       const double b, const double *const yy, double *const zz);

}
