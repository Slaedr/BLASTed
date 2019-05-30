/** \file
 * \brief Computation of adjacency lists of matrices
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

#include "adjacency.hpp"

namespace blasted {

template <typename scalar, typename index>
ColumnAdjacency<scalar,index>::ColumnAdjacency(const CRawBSRMatrix<scalar,index>& mat)
{
	ptrs.resize(mat.nbrows+1);
	rows_nz.resize(mat.browind[mat.nbrows]);
	rows_loc.resize(mat.browind[mat.nbrows]);

	index iz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++) {
		}
	}
}

template class ColumnAdjacency<double,int>;

}
