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

#include "helper_algorithms.hpp"
#include "adjacency.hpp"

namespace blasted {

template <typename scalar, typename index>
ColumnAdjacency<scalar,index>::ColumnAdjacency(const CRawBSRMatrix<scalar,index>& mat)
{
	compute(mat);
}

template <typename scalar, typename index>
void ColumnAdjacency<scalar,index>::compute(const CRawBSRMatrix<scalar,index>& mat)
{
	ptrs.assign(mat.nbrows+1, 0);
	col_rows.resize(mat.browptr[mat.nbrows]);
	rows_loc.resize(mat.browptr[mat.nbrows]);

	// first determine size of each column while leaving the first entry of ptrs empty
	for(index jj = 0; jj < mat.browptr[mat.nbrows]; jj++) {
		ptrs[mat.bcolind[jj]+1]++;
	}
	if(ptrs.front() != 0)
		throw std::runtime_error("Wrong sizes!");

	// compute beginnings of data list for each column
	internal::inclusive_scan(ptrs);
	if(ptrs.back() != mat.browptr[mat.nbrows])
		throw std::runtime_error("Wrong pointer list!");

	std::vector<index> colfill(mat.nbrows, 0);

	// Fill required indices (serially)
	for(index irow = 0; irow < mat.nbrows; irow++)
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			const index colind = mat.bcolind[jj];

			col_rows[ptrs[colind]+colfill[colind]] = irow;
			rows_loc[ptrs[colind]+colfill[colind]] = jj;

			colfill[colind]++;
		}
}

template class ColumnAdjacency<double,int>;

}
