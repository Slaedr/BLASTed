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
	if(rp.size() > 0) {
		// move rows around
		std::vector<scalar> tempval(mat.browptr[mat.nbrows]*bs);
		std::vector<index> tempcind(mat.browptr[mat.nbrows]);
		std::vector<index> temprptr(mat.nbrows+1);
		temprptr[mat.nbrows] = mat.browptr[mat.nbrows];

		index pos = 0;

		for(index i = 0; i < mat.nbrows; i++) {
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

	if(cp.size() > 0) {
		// move columns around
#pragma omp parallel for default(shared) schedule(dynamic,100)
		for(index i = 0; i < mat.nbrows; i++) {
			// switch the column indices
			for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				mat.bcolind[jj] = cp[mat.bcolind[jj]];
			// re-sort
			internal::sortBlockInnerDimension<scalar,index,bs>(&mat.bcolind[mat.browptr[i]],
			                                                   &mat.vals[mat.browptr[i]*bs*bs]);
		}
	}
}

template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::applyOrdering(scalar *const vec,
                                                const RSApplyMode mode, const RSApplyDir dir) const
{
	if(mode == FORWARD) {
		if(dir == ROW) {
			assert(rp.size() > 0);
			// apply row ordering
		}
		else {
			assert(cp.size() > 0);
			// apply column ordering
		}
	}
	else {
		if(dir == ROW) {
			assert(rp.size() > 0);
			// apply inverse row ordering
		}
		else {
			assert(cp.size() > 0);
			// apply inverse column ordering
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

