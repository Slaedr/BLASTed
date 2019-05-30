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
	ptrs.assign(mat.nbrows+1, 0);
	rows_nz.resize(mat.browind[mat.nbrows]);
	rows_loc.resize(mat.browind[mat.nbrows]);

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

}

template class ColumnAdjacency<double,int>;

}
