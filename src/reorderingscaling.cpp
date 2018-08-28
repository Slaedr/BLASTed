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
void Reordering<scalar,index,bs>::applyOrdering(RawBSRMatrix<scalar,index>& mat) const
{
	if(rp.size() > 0)
	{
		// move rows around
		std::vector<scalar> tempval(mat.browptr[mat.nbrows]*bs);
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
#pragma omp parallel for default(shared) schedule(dynamic,100)
		for(index i = 0; i < mat.nbrows; i++)
		{
			scalar *const mvals = &mat.vals[mat.browptr[i]*bs*bs];
			index *const mcolind = &mat.bcolind[mat.browptr[i]];
			const index browsz = (mat.browptr[i+1]-mat.browptr[i]);

			std::vector<index> cind(browsz);

			// copy the column indices into a temporary location
			for(index jj = 0; jj < browsz; jj++)
				cind[jj] = mcolind[jj];

			// Change column indices to reflect the new ordering
			for(index jj = 0; jj < browsz; jj++)
			{
				const index pcind = cp[cind[jj]];
				auto it = std::find(cind.begin(), cind.end(), pcind);
				const index pos = it - cind.begin();
				mcolind[pos] = cind[jj];
			}

			// Sort both the column indices and the non-zero values according to the column indices
			internal::sortBlockInnerDimension<scalar,index,bs>(browsz, mcolind, mvals);
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

	if(mode == FORWARD) {

		if(dir == ROW) {

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

		if(dir == ROW) {

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

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index,bs>::applyScaling(RawBSRMatrix<scalar,index>& mat)
{
	if(rowscale.size() > 0) {
		// scale rows
	}
	if(colscale.size() > 0) {
		// scale columns
	}
}

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index>::applyScaling(scalar *const vec, const RSApplyMode mode,
                                                   const RSApplyDir dir) const
{
	if(mode == FORWARD) {
		if(dir == ROW) {
			assert(rowscale.size() > 0);
			// apply row scaling
		}
		else {
			assert(colscale.size() > 0);
			// apply column scaling
		}
	}
	else {
		if(dir == ROW) {
			assert(rowscale.size() > 0);
			// apply inverse row scaling
		}
		else {
			assert(colscale.size() > 0);
			// apply inverse column scaling
		}
	}
}

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

