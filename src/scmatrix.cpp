/** \file
 * \brief Implementation of some methods related to sparse-column storage of matrices
 * \author Aditya Kashi
 * \date 2018-08
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

#include "scmatrixdefs.hpp"

namespace blasted {

template <typename scalar, typename index, int bs>
RawBSCMatrix<scalar,index> convert_BSR_to_BSC(const CRawBSRMatrix<scalar,index> *const rmat)
{
	static_assert(bs == 1, "Block version of conversion to BSC not implemented yet!");

	RawBSCMatrix<scalar,index> cmat;
	const index bnnz = rmat.browptr[rmat.nbrows];
	cmat.nbcols = rmat.nbrows;
	cmat.bcolptr = new index[cmat.nbcols+1];
	cmat.browind = new index[bnnz];
	cmat.vals = new scalar[bnnz];
	cmat.diagind = new index[cmat.nbcols];

	// TODO

	return cmat;
}

template RawBSCMatrix<double,int>
convert_BSR_to_BSC<double,int,1>(const CRawBSRMatrix<double,int> *const rmat);

template <typename scalar, typename index>
void destroyRawBSCMatrix(RawBSCMatrix<scalar,index>& mat)
{
	delete [] mat.bcolptr;
	delete [] mat.browind;
	delete [] mat.vals;
	delete [] mat.diagind;
	mat.nbcols = 0;
}

template void destroyRawBSCMatrix<double,int>(RawBSCMatrix<double,int>& mat);

}
