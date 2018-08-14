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

#include "reorderingscaling.hpp"

namespace blasted {

template <typename scalar, typename index>
Reordering<scalar,index>::Reordering()
{ }

template <typename scalar, typename index>
Reordering<scalar,index>::~Reordering()
{ }

template <typename scalar, typename index>
Reordering<scalar,index>::setOrdering(const index *const rord, const index *const cord,
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

