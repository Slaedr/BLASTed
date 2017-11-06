/** \file coomatrix.hpp
 * \brief Specifies coordinate matrix formats
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

#ifndef COOMATRIX_H
#define COOMATRIX_H

#include <cassert>
#include <vector>
#include <string>
#include <boost/algorithm/string/split.hpp>
#include "blockmatrices.hpp"

namespace blasted {

enum MMMatrixType {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};
enum MMStorageType {COORDINATE, ARRAY};
enum MMScalarType {REAL, COMPLEX, INTEGER, PATTERN};

/// Information contained in the Matrix Market format's first line
/** A description of the format is given [here](http://math.nist.gov/MatrixMarket/formats.html).
 */
struct MMDescription {
	MMStorageType storagetype;
	MMScalarType scalartype;
	MMMatrixType matrixtype;
};

/// A sparse matrix with entries stored in coordinate format
template <typename scalar, typename index>
class COOMatrix : public AbstractMatrix<scalar,index>
{
public:
	COOMatrix(const char storagetype) : AbstractMatrix<scalar,index>(storagetype)
	{ }

	~COOMatrix();

	/// Reads a matrix from a file in Matrix Market format
	void readMatrixMarket(const std::string file);
	
	void setStructure(const index n, const index *const vec1, const index *const vec2);
	
	void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	void convertToCSR(BSRMatrix<scalar,index,1> *const cmat) const;

	template<int bs>
	void convertToBSR(BSRMatrix<scalar,index,bs> *const bmat) const;

protected:

	/// Returns a [description](\ref MMDescription) of the matrix if it's in Matrix Market format
	MMDescription getMMDescription(std::ifstream& fin);

	std::vector<scalar> vals;           ///< Stored entries of the matrix
	std::vector<index> rowinds;        	///< Row indices of stored values
	std::vector<index> colinds;         ///< Column indices of stored values
};

#include "coomatrix.ipp"

}

#endif
