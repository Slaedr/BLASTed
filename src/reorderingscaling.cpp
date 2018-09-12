/** \file
 * \brief Implementation of schemes to reorder and scale matrices for various purposes
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

#include <algorithm>
#include <utility>
#include "helper_algorithms.hpp"
#include "reorderingscaling.hpp"

namespace blasted {

template <typename scalar, typename index, int bs>
Reordering<scalar,index,bs>::Reordering()
{ }

template <typename scalar, typename index, int bs>
Reordering<scalar,index,bs>::~Reordering()
{ }

template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::setOrdering(const index *const rord, const index *const cord,
                                              const index length)
{
	if(rord) {
		rp.resize(length);
		for(index i = 0; i < length; i++)
			rp[i] = rord[i];
	}
	if(cord) {
		cp.resize(length);
		for(index i = 0; i < length; i++)
			cp[i] = cord[i];
	}
}

template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::applyOrdering(RawBSRMatrix<scalar,index>& mat,
                                                const RSApplyMode mode) const
{
	if(mode == FORWARD) {
		if(rp.size() > 0)
		{
			// move rows around
			std::vector<scalar> tempval(mat.browptr[mat.nbrows]*bs*bs);
			std::vector<index> tempcind(mat.browptr[mat.nbrows]);
			std::vector<index> temprptr(mat.nbrows+1);
			temprptr[mat.nbrows] = mat.browptr[mat.nbrows];

			index pos = 0;

			for(index i = 0; i < mat.nbrows; i++)
			{
				const index ni = rp[i];
				temprptr[i] = pos;
				for(index jj = mat.browptr[ni]; jj < mat.browptr[ni+1]; jj++) {
					tempcind[pos] = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						tempval[pos*bs*bs + k] = mat.vals[jj*bs*bs+k];
					pos++;
				}
			}

			assert(pos == mat.browptr[mat.nbrows]);

			// copy into original array
			for(index i = 0; i < mat.nbrows; i++)
				mat.browptr[i] = temprptr[i];
			for(index jj = 0; jj < pos; jj++) {
				mat.bcolind[jj] = tempcind[jj];
				for(int k = 0; k < bs*bs; k++)
					mat.vals[jj*bs*bs+k] = tempval[jj*bs*bs + k];
			}
		}

		if(cp.size() > 0)
		{
			// move columns around
			//#pragma omp parallel for default(shared) schedule(dynamic,200)
			for(index i = 0; i < mat.nbrows; i++)
			{
				scalar *const rvals = &mat.vals[mat.browptr[i]*bs*bs];
				index *const rcolind = &mat.bcolind[mat.browptr[i]];
				const index rowsz = (mat.browptr[i+1]-mat.browptr[i]);

				std::vector<index> cind(rcolind, rcolind+rowsz);

				// Change column indices to reflect the new ordering
				for(index jj = 0; jj < rowsz; jj++)
				{
					const index pcind = cp[cind[jj]];
					auto it = std::find(cind.begin(), cind.end(), pcind);

					// If the transformed column index was not found, the new matrix has a zero
					//  at the current column index (cind[jj]).
					if(it == cind.end())
						continue;

					// If the transformed column index is found, the value at that index
					//  needs to move to the current column index cind[jj].
					const index pos = it - cind.begin();
					rcolind[pos] = cind[jj];
				}

				// Sort both the column indices and the non-zero values according to the column indices
				internal::sortBlockInnerDimension<scalar,index,bs>(rowsz, rcolind, rvals);
			}
		}
	}
	else {
		// inverse ordering

		if(rp.size() > 0)
		{
			// copy the matrix to a list of rows
			using CSEntry = std::pair<index,scalar[bs*bs]>;
			std::vector<std::vector<CSEntry>> brows(mat.nbrows);

			// can be parallelized
			for(index i = 0; i < mat.nbrows; i++)
			{
				// Row rp[i] in the permuted matrix gets the i-th row of the original matrix
				const index destr = rp[i];
				assert(brows[destr].size() == 0);
				brows[destr].resize(mat.browptr[i+1]-mat.browptr[i]);

				for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				{
					const index dj = jj-mat.browptr[i];
					brows[destr][dj].first = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						brows[destr][dj].second[k] = mat.vals[jj*bs*bs+k];
				}
			}

			// copy back
			mat.browptr[0] = 0;
			for(index i = 0; i < mat.nbrows; i++)
			{
				mat.browptr[i+1] = static_cast<index>(brows[i].size()) + mat.browptr[i];
				for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				{
					const index dj = jj-mat.browptr[i];
					mat.bcolind[jj] = brows[i][dj].first;
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs+k] = brows[i][dj].second[k];
				}
			}
		}

		if(cp.size() > 0)
		{
			// rename columns
			//#pragma omp parallel for default(shared) schedule(dynamic,200)
			for(index i = 0; i < mat.nbrows; i++)
			{
				index *const rcolind = &mat.bcolind[mat.browptr[i]];
				scalar *const rvals = &mat.vals[mat.browptr[i]*bs*bs];
				const index rsize = mat.browptr[i+1]-mat.browptr[i];

				// copy column indices of this row into a temp vector
				const std::vector<index> ocinds(rcolind, rcolind+rsize);

				// transform column indices with the forward permutation, so that
				//  the actual matrix is transformed with the inverse permutation.
				for(index jj = 0; jj < rsize; jj++)
					rcolind[jj] = cp[ocinds[jj]];

				internal::sortBlockInnerDimension<scalar,index,bs>(rsize, rcolind, rvals);
			}
		}
	}
}

/** This is most likely not the best way to do it. The vector is first copied into a local
 * temporary storage which is freed in the end.
 */
template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::applyOrdering(scalar *const vec,
                                                const RSApplyMode mode, const RSApplyDir dir) const
{
	const index size = dir == ROW ? rp.size() : cp.size();
	assert(size > 0);

	// copy vector to temp location
	std::vector<scalar> tv(size*bs);
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < size*bs; i++)
		tv[i] = vec[i];

	if(mode == FORWARD)
	{
		if(dir == ROW)
		{
			// apply row ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[i*bs+k] = tv[rp[i]*bs+k];
		}
		else {
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[i*bs+k] = tv[cp[i]*bs+k];
		}
	}
	else {
		if(dir == ROW)
		{
			// apply inverse row ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[rp[i]*bs+k] = tv[i*bs+k];
		}
		else {
			// apply inverse column ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[cp[i]*bs+k] = tv[i*bs+k];
		}
	}
}

template class ReorderingScaling<double,int,1>;
template class ReorderingScaling<double,int,4>;
template class ReorderingScaling<double,int,7>;

template <typename scalar, typename index, int bs>
ReorderingScaling<scalar,index,bs>::ReorderingScaling()
	: Reordering<scalar,index,bs>()
{ }

template <typename scalar, typename index, int bs>
ReorderingScaling<scalar,index,bs>::~ReorderingScaling()
{ }

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index,bs>::applyScaling(RawBSRMatrix<scalar,index>& mat,
                                                      const RSApplyMode mode) const
{
	if(rowscale.size() > 0)
	{
		// scale rows
		if(mode == FORWARD) {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] *= rowscale[irow];
			}
		}
		else {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] /= rowscale[irow];
			}
		}
	}

	if(colscale.size() > 0)
	{
		// scale columns
		if(mode == FORWARD) {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++) {
					const index column_index = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] *= colscale[column_index];
				}
			}
		}
		else {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++) {
					const index column_index = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] /= colscale[column_index];
				}
			}
		}
	}
}

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index,bs>::applyScaling(scalar *const vec, const RSApplyMode mode,
                                                      const RSApplyDir dir) const
{
	if(mode == FORWARD)
	{
		if(dir == ROW)
		{
			if(rowscale.size() > 0)
				// apply row scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(rowscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] *= rowscale[i];
		}
		else
		{
			if(colscale.size() > 0)
				// apply column scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(colscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] *= colscale[i];
		}
	}
	else {
		if(dir == ROW)
		{
			if(rowscale.size() > 0)
				// apply inverse row scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(rowscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] /= rowscale[i];
		}
		else {
			if(colscale.size() > 0)
				// apply inverse column scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(colscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] /= colscale[i];
		}
	}
}

template class Reordering<double,int,1>;
template class Reordering<double,int,4>;
template class Reordering<double,int,7>;

}

#ifdef HAVE_MC64
extern "C" {

extern void mc64a_(const int *const job, const int *const n, const int *const ne,
                   const int *const colptr, const int *const rowind, const float *const A,
                   int *const num_diag, int *const cperm,
                   const int *const len_workvec, int *const workvec,
                   const int *const len_scalevec, float *const scalevec,
                   int icntl[10], int info[10]);

extern void mc64ad_(const int *const job, const int *const n, const int *const ne,
                    const int *const colptr, const int *const rowind, const double *const A,
                    int *const num_diag, int *const cperm,
                    const int *const len_workvec, int *const workvec,
                    const int *const len_scalevec, double *const scalevec,
                    int icntl[10], int info[10]);
}

namespace blasted {

}

#endif

