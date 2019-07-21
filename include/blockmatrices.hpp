/** \file blockmatrices.hpp
 * \brief Classes for sparse square matrices consisting of square dense blocks of static size
 *
 * There are two kinds of sparse matrix classes. One kind, consisting of \ref BSRMatrix,
 * implements a matrix which provides assembling operations and allocates and handles its own data.
 * The other kind, \ref BSRMatrixView, only provides BLAS 2 operations while wrapping an
 * external matrix. It does not allocate or delete storage for the matrix itself.
 *
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef BLASTED_BLOCKMATRICES_H
#define BLASTED_BLOCKMATRICES_H

#include <iostream>
#include <limits>

#include "linearoperator.hpp"
#include "srmatrixdefs.hpp"
#include "reorderingscaling.hpp"

namespace blasted {

/// An abstract matrix view used for wrapping matrices with compressed (block-)sparse row storage
template<typename scalar, typename index>
class SRMatrixView : public MatrixView<scalar, index>
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");

public:
	/// A constructor which just wraps a compressed BSR matrix described by 4 arrays
	/** \param[in] n_brows Number of block-rows
	 * \param[in] brptrs Array of block-row pointers
	 * \param[in] bcinds Array of block-column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of pointers to diagonal blocks
	 * \param[in] storagetype The type of sparse representation of the matrix data
	 */
	SRMatrixView(const index n_brows, const index *const brptrs, const index *const bcinds,
	             const scalar *const values, const index *const diaginds, const StorageType storagetype);

	/// Construct from a SRMatrixStorage
	SRMatrixView(SRMatrixStorage<const scalar,const index>&& srmat, const StorageType storagetype);

	/// Allows immutable access to the underlying matrix storage
	const SRMatrixStorage<const scalar,const index>& getSRStorage() const {
		return mat;
	}

protected:

	/// The SR matrix wrapper
	SRMatrixStorage<const scalar,const index> mat;
};

/// A BSR matrix that is formed by wrapping a pre-existing read-only matrix
/** StorageOptions is an Eigen type describing storage options, which we use here to
 * specify whether storage within blocks is row-major or column-major. If each block is row-major,
 * stopt should be Eigen::RowMajor else it should be Eigen::ColMajor.
 * The blocks are always arranged block-row-wise relative to each other.
 */
template <typename scalar, typename index, int bs, StorageOptions stopt>
class BSRMatrixView : public SRMatrixView<scalar, index>
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stopt == RowMajor || stopt == ColMajor, "Invalid storage option!");

public:
	/// A constructor which just wraps a BSR matrix described by 4 arrays
	/** \param[in] n_brows Number of block-rows
	 * \param[in] brptrs Array of block-row pointers
	 * \param[in] bcinds Array of block-column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of pointers to diagonal blocks
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrixView(const index n_brows, const index *const brptrs,
	              const index *const bcinds, const scalar *const values, const index *const dinds);

	/// Construct from a SRMatrixStorage
	BSRMatrixView(SRMatrixStorage<const scalar,const index>&& srmat);

	/// Cleans up temporary data needed for preconditioning operations
	virtual ~BSRMatrixView();

	/// Computes the matrix vector product of this matrix with one vector-- y := Ax
	virtual void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	/** \warning x must not alias z.
	 */
	virtual void gemv3(const scalar a, const scalar *const __restrict x,
	                   const scalar b, const scalar *const y,
	                   scalar *const z) const;

	/// Returns the dimension (number of rows) of the square matrix
	index dim() const { return mat.nbrows*bs; }

protected:

	using SRMatrixView<scalar,index>::mat;
};

/// A CSR matrix formed by wrapping a read-only matrix
/**
 * On destruct, cleans up only its own data that are needed for preconditioning operations.
 */
template <typename scalar, typename index>
class CSRMatrixView : public SRMatrixView<scalar, index>
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");

public:

	/// A constructor which wraps a CSR matrix described by 4 arrays
	/** \param[in] nrows Number of rows
	 * \param[in] rptrs Array of row pointers
	 * \param[in] cinds Array of column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of diagonal entry pointers
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	CSRMatrixView(const index nrows, const index *const rptrs,
	              const index *const cinds, const scalar *const values, const index *const dinds);

	/// Construct from a SRMatrixStorage
	CSRMatrixView(SRMatrixStorage<const scalar,const index>&& srmat);

	/// De-allocates temporary storage only, not the matrix storage itself
	virtual ~CSRMatrixView();

	/// Computes the matrix vector product of this matrix with one vector-- y := Ax
	virtual void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
	                   const scalar b, const scalar *const y,
	                   scalar *const z) const;
	
	/// Returns the number of rows in the matrix
	index dim() const { return mat.nbrows; }

protected:
	
	/// The CSR matrix data	
	using SRMatrixView<scalar,index>::mat;
};

/// Block sparse row matrix
/** Scalars inside each dense block are stored in a row-major order.
 * Unsigned types cannot be used as index; that will cause SGS and ILU preconditioners to fail.
 */
template <typename scalar, typename index, int bs>
class BSRMatrix : public AbstractMatrix<scalar, index>
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(bs > 0, "Block size must be positive!");

public:
	/// Initialization without pre-allocation of any storage
	BSRMatrix();

	/// Deep copy another BSR matrix of the same block-size
	BSRMatrix(const BSRMatrix<scalar,index,bs>& mat);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply (deep) copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply (deep) copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs);
	
	/// A constructor which just wraps a BSR matrix described by 4 arrays
	/** \param[in] n_brows Number of block-rows
	 * \param[in] brptrs Array of block-row pointers
	 * \param[in] bcinds Array of block-column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of diagonal entry pointers
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrix(const index n_brows, index *const brptrs,
	          index *const bcinds, scalar *const values, index *const dinds);

	/// Transfers the arrays of a raw BSR matrix to itself and nulls the raw matrix
	/** Make sure that memory in rmat is allocated using Boost aligned_alloc, because it will be
	 * freed under that assumption.
	 */
	BSRMatrix(RawBSRMatrix<scalar,index>& rmat);

	/// De-allocates memory
	virtual ~BSRMatrix();

	/// Allows immutable access to the underlying matrix storage
	const CRawBSRMatrix<scalar,index> *getRawSRMatrix() const {
		return reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat);
	}

	/// Set the storage structure of the matrix
	/** \param[in] nbrows Number of block-rows
	 * \param[in] bcinds Array of block-column indices for all non-zero blocks
	 * \param[in] brptrs Array of indices into bcinds pointing to where each block-row starts
	 *
	 * \warning Deletes pre-existing contents!
	 */
	void setStructure(const index nbrows, const index *const bcinds, const index *const brptrs);

	/// Sets all stored entries of the matrix to zero
	void setAllZero();

	/// Sets diagonal blocks to zero
	void setDiagZero();
	
	/// Insert a block of values into the [matrix](\ref vals); not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location of the matrix at the same time.
	 * \param[in] starti The *row index* (not block-row index) at which the block to be added starts
	 * \param[in] startj The *column index* (not block-column index) at which the block starts
	 * \param[in] buffer The entries of the block
	 * \param[in] param1 Dummy, can be set to any value; not used
	 * \param[in] param2 Dummy, can be set to any value; not used
	 */
	void submitBlock(const index starti, const index startj, 
			const scalar *const buffer, const index param1, const index param2);

	/// Update a (contiguous) block of values into the [matrix](\ref vals)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 * \param[in] starti The *row index* (not block-row index) at which 
	 *              the block to be updated starts
	 * \param[in] startj The *column index* (not block-column index) at which the block starts
	 * \param[in] buffer The entries of the block
	 * \param[in] param1 Dummy, can be set to any value; not used
	 * \param[in] param2 Dummy, can be set to any value; not used
	 */
	void updateBlock(const index starti, const index startj, 
			const scalar *const buffer, const index param1, const index param2);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The *row index* at which the  diagonal block to be updated starts
	 * \param[in] buffer The values making up the block to be added
	 * \param[in] param Dummy, can be set to any value; not used
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer, const index param);

	/// Scales the matrix by a scalar
	void scaleAll(const scalar factor);

	/// Computes the matrix vector product of this matrix with one vector-- y := Ax
	virtual void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	/** \warning x must not alias z.
	 */
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
	                   const scalar b, const scalar *const y,
	                   scalar *const z) const;

	/// Compute some ordering and/or scaling using this matrix
	void computeOrderingScaling(ReorderingScaling<scalar,index,bs>& rs) const;

	/// Apply some ordering/scaling to this matrix
	/** First applies the scaling, and then the ordering.
	 */
	void reorderScale(const ReorderingScaling<scalar,index,bs>& rs, const RSApplyMode mode);

	/// Checks equality with another BSR matrix
	/** \return Returns 5 booleans corresponding to equality of, in order,
	 * - Number of block-rows
	 * - Block-row pointers (including total number of non-zero blocks)
	 * - Block-column indices
	 * - Non-zero values
	 * - Positions of diaginal blocks
	 */
	std::array<bool,5> isEqual(const BSRMatrix<scalar,index,bs>& other, const scalar tol) const;
	
	/// Returns the dimension (number of rows) of the square matrix
	index dim() const { return mat.nbrows*bs; }

protected:

	/// The BSR matrix storage
	SRMatrixStorage<scalar,index> mat;
};

/// Compressed sparse row (CSR) matrix
/** The limiting case of BSR matrix when block size is 1
 */
template <typename scalar, typename index>
class BSRMatrix<scalar,index,1> : public AbstractMatrix<scalar, index>
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");

public:

	/// Minimal initialzation; just sets number of async sweeps
	BSRMatrix();

	/// Deep copy another CSR matrix
	BSRMatrix(const BSRMatrix<scalar,index,1>& mat);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of rows
	 * \param[in] bcinds Column indices (copied)
	 * \param[in] brptrs Row pointers (copied)
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs);

	/// A constructor which just wraps a CSR matrix described by 4 arrays
	/** \param[in] nrows Number of rows
	 * \param[in] rptrs Array of row pointers
	 * \param[in] cinds Array of column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of diagonal entry pointers
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrix(const index nrows, index *const rptrs,
	          index *const cinds, scalar *const values, index *const dinds);

	/// Transfers the arrays of a raw CSR matrix to itself and nulls the raw matrix
	BSRMatrix(RawBSRMatrix<scalar,index>& rmat);

	/// De-allocates memory
	virtual ~BSRMatrix();

	/// Allows immutable access to the underlying matrix storage
	const SRMatrixStorage<scalar,index>& getRawSRMatrix() const {
		return mat;
	}

	/// Set the storage structure of the matrix
	/** \param[in] nbrows Number of rows
	 * \param[in] bcinds Array of column indices for all non-zero entries
	 * \param[in] brptrs Array of indices into bcinds pointing to where each row starts
	 *
	 * \waring Deletes pre-existing contents!
	 */
	void setStructure(const index nbrows, const index *const bcinds, const index *const brptrs);

	/// Sets all stored entries of the matrix to zero
	void setAllZero();

	/// Sets diagonal blocks to zero
	void setDiagZero();

	/// Insert a block of values into the [matrix](\ref vals); not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location of the matrix at the same time.
	 * \param[in] starti The (global) row index of the first row of the block to be inserted
	 * \param[in] startj The (global) column index of the first row of the block to be inserted
	 * \param[in] bsi The number of rows in the block being inserted
	 * \param[in] bsj The number of columns in the block being inserted
	 */
	void submitBlock(const index starti, const index startj,
	                 const scalar *const buffer, const index bsi, const index bsj);

	/// Update a (contiguous) block of values into the [matrix](\ref vals)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 * \param[in] starti The (global) row index of the first row of the block to be inserted
	 * \param[in] startj The (global) column index of the first row of the block to be inserted
	 * \param[in] bsi The number of rows in the block being inserted
	 * \param[in] bsj The number of columns in the block being inserted
	 */
	void updateBlock(const index starti, const index startj,
	                 const scalar *const buffer, const index bsi, const index bsj);

	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The row whose diagonal block is to be updated
	 * \param[in] buffer The values making up the block to be added
	 * \param[in] bs Size of the square block to be updated
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer, const index bs);

	/// Scales the matrix by a scalar
	void scaleAll(const scalar factor);

	/// Computes the matrix vector product of this matrix with one vector-- y := Ax
	virtual void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x,
	                   const scalar b, const scalar *const y,
	                   scalar *const z) const;

	/// Returns the number of rows in the matrix
	index dim() const { return mat.nbrows; }

	/// Compute some ordering and/or scaling using this matrix
	void computeOrderingScaling(ReorderingScaling<scalar,index,1>& rs) const;

	/// Apply some ordering/scaling to this matrix
	/** First applies the scaling, and then the ordering.
	 */
	void reorderScale(const ReorderingScaling<scalar,index,1>& rs, const RSApplyMode mode);

	/// Checks equality with another CSR matrix
	/** \return Returns 5 booleans corresponding to equality of, in order,
	 * - Number of rows
	 * - row pointers (including total number of non-zero blocks)
	 * - column indices
	 * - Non-zero values
	 * - Positions of diaginal entries
	 */
	std::array<bool,5> isEqual(const BSRMatrix<scalar,index,1>& other, const scalar tol) const;

	/// Returns the first row that has a zero diagonal, -1 if no row has zero diagonal
	index zeroDiagonalRow() const;

	size_t getNumZeroDiagonals() const;

	/// Returns the product of diagonal entries of the matrix
	scalar getDiagonalProduct() const;

	/// Returns the sum of the absolute values of diagonal entries of the matrix
	scalar getDiagonalAbsSum() const;

	/// Returns the minimum entry on the diagonal
	scalar getAbsMinDiagonalEntry() const;

	/// Returns the maximum of absolute values of diagonal entries
	scalar getAbsMaxDiagonalEntry() const;

protected:

	/** Indicates whether the data (\ref vals, \ref bcolind, \ref browptr, \ref diagind)
	 * is owned by this object
	 */
	bool owner;

	/// The CSR matrix data
	SRMatrixStorage<scalar,index> mat;
};

}
#endif
