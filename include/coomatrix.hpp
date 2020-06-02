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

#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#include "device_container.hpp"
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

/// Reads a dense matrix from a Matrix Market file in row-major format
template <typename scalar>
device_vector<scalar> readDenseMatrixMarket(const std::string file);

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
	static_assert(!std::is_const<index>::value, "Index type should be mutable.");
	static_assert(!std::is_const<scalar>::value, "Scalar type should be mutable.");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");

public:
	COOMatrix();

	virtual ~COOMatrix();

	/// Returns the total number of rows
	index numrows() const;
	/// Total number of columns
	index numcols() const;
	/// Total number of non-zero entries
	index numnonzeros() const;

	/// Reads a matrix from a file in Matrix Market format
	/** We store the row and column indices with zero-based indexing, but
	 * the file contains 1-based indexing.
	 */
	void readMatrixMarket(const std::string file);

	/// Creates a new sparse-row matrix from the COO matrix
	/** The returned matrix owns its storage and frees it when destroyed.
	 * Member nbrows of the SRMatrixStorage is set to the total number of rows.
	 */
	SRMatrixStorage<scalar,index> convertToCSR() const;

	/// Creates a new block sparse-row matrix from the COO matrix
	/** The block size is given by the template parameter bs.
	 * The template parameter stor specifies whether the scalars within a block are stored
	 * row-major or column-major.
	 * The returned matrix owns its storage and frees it when destroyed.
	 * SRMatrixStorage::nbrows is set to the number of block-rows.
	 */
	template<int bs, StorageOptions stor>
	SRMatrixStorage<scalar,index> convertToBSR() const;

	/// Get immutable access to entries of the matrix
	const std::vector<Entry<scalar,index>>& getEntries() const;

	/// Get immutable access to row pointers' array
	const std::vector<index>& getRowPtrs() const;

protected:

	std::vector<Entry<scalar,index>> entries;     ///< Stored entries of the matrix
	index nnz;                                    ///< Number of nonzeros
	index nrows;                                  ///< Number of rows
	index ncols;                                  ///< Number of columns

	std::vector<index> rowptr;                    ///< Vector of row pointers into \ref entries
};

/// Build a BSR (row-major blocks) or CSR matrix from a Matrix Market file in COO format
template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs> constructBSRMatrixFromMatrixMarketFile(const std::string file);

/// Build a BSR or CSR matrix from a Matrix Market file in COO format
template <typename scalar, typename index, int bs>
SRMatrixStorage<scalar,index> getSRMatrixFromCOO(const COOMatrix<scalar,index>& coo_mat,
                                                 const std::string block_storage_order);

/// Exception thrown if the matrix (or vector) file is incorrect or unsupported
class MatrixReadException : public std::runtime_error
{
public:
	MatrixReadException(const std::string& msg);
};

}

#endif
