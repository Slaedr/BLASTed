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
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "blockmatrices.hpp"

namespace blasted {

/// Encodes matrix types in matrix market file format
enum MMMatrixType {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};
/// Encodes matrix storage type in a matrix market file
enum MMStorageType {COORDINATE, ARRAY};
/// Encodes the scalar type of a matrix in a matrix market file
enum MMScalarType {REAL, COMPLEX, INTEGER, PATTERN};

/// Information contained in the Matrix Market format's first line
/** A description of the format is given [here](http://math.nist.gov/MatrixMarket/formats.html).
 */
struct MMDescription {
	MMStorageType storagetype;
	MMScalarType scalartype;
	MMMatrixType matrixtype;
};

/// Returns a [description](\ref MMDescription) of the matrix if it's in Matrix Market format
MMDescription getMMDescription(std::ifstream& fin);

/// Returns a vector containing size information of a matrix in a Matrix Market file
template <typename index>
std::vector<index> getSizeFromMatrixMarket(
		std::ifstream& fin,                   ///< Opened file stream to read from
		const MMDescription& descr            ///< Matrix description
	);

/// Reads a dense matrix from a Matrix Market file in row-major format
/** Allocates the required storage and returns a pointer to it.
 */
template <typename scalar>
scalar * readDenseMatrixMarket(const std::string file);

/// A triplet that encapsulates one entry of a coordinate matrix
template <typename scalar, typename index>
struct Entry {
	index rowind;
	index colind;
	scalar value;
};

/// A sparse matrix with entries stored in coordinate format
template <typename scalar, typename index>
class COOMatrix
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");

public:
	COOMatrix();

	virtual ~COOMatrix();

	/// Reads a matrix from a file in Matrix Market format
	/** We store the row and column indices with zero-based indexing, but
	 * the file contains 1-based indexing.
	 */
	void readMatrixMarket(const std::string file);
	
	// Multiplies the matrix with a vector
	//void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Converts to a compressed sparse row matrix
	void convertToCSR(BSRMatrix<scalar,index,1> *const cmat) const;

	/// Converts to a raw CSR struct \ref RawBSRMatrix
	/** Member nbrows of RawBSRMatrix is set to the total number of rows.
	 */
	void convertToCSR(RawBSRMatrix<scalar,index> *const cmat) const;

	/// Converts to a compressed sparse block-row (BSR) matrix 
	/** The block size is given by the template parameter bs.
	 * The template parameter stor specifies whether the scalars within a block are stored
	 * row-major or column-major.
	 * The storage required for the matrix is allocated here in the arrays of bmat.
	 * RawBSRMatrix::nbrows is set to the number of block-rows.
	 */
	template<int bs, StorageOptions stor>
	void convertToBSR(RawBSRMatrix<scalar,index> *const bmat) const;

protected:

	std::vector<Entry<scalar,index>> entries;     ///< Stored entries of the matrix
	index nnz;                                    ///< Number of nonzeros
	index nrows;                                  ///< Number of rows
	index ncols;                                  ///< Number of columns

	std::vector<index> rowptr;                    ///< Vector of row pointers into \ref entries
};

#include "coomatrix.ipp"

}

#endif
