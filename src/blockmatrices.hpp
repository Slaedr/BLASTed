/** \file blockmatrices.hpp
 * \brief Classes for sparse matrices consisting of square dense blocks of static size
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef BLOCKMATRICES_H
#define BLOCKMATRICES_H

// get around dependent templates for Eigen
#define SEG template segment
#define BLK template block

#include "linearoperator.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/LU>

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// Block sparse row matrix
/** Dense blocks stored in a (block-) row-major storage order.
 * Unsigned types cannot be used as index; that will cause SGS and ILU preconditioners to fail.
 */
template <typename scalar, typename index, int bs>
class BSRMatrix : public LinearOperator<scalar, index>
{
public:
	/// Initialization without pre-allocation of any storage
	BSRMatrix(const short nbuildsweeps, const short napplysweeps);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply (deep) copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply (deep) copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const short nbuildsweeps, const short napplysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();

	/// Set the storage structure of the matrix
	/** \param[in] nbrows Number of block-rows
	 * \param[in] bcinds Array of block-column indices for all non-zero blocks
	 * \param[in] brptrs Array of indices into bcinds pointing to where each block-row starts
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

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner. 
	/** Approximately solves D z = r
	 */
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Allocates storage for a vector \ref ytemp required for both SGS and ILU applications
	void allocTempVector();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	/** Approximately solves (D+L) D^(-1) (D+U) z = r
	 * where D, L and U are the diagonal, upper and lower parts of the matrix respectively.
	 */
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	/** Finds \f$ \tilde{L} \f$ and \f$ \tilde{U} \f$ such that
	 * \f[ \tilde{L} \tilde{U} = A \f]
	 * is approximately satisifed, with \f$ \tilde{L} \f$ unit lower triangular.
	 * The sparsity if A is preserved, so this is ILU(0).
	 */
	void precILUSetup();

	/// Applies a block LU factorization L U z = r
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the dimension (number of rows) of the square matrix
	index dim() const { return nbrows*bs; }

	// Prints the matrix out to a file in dense format
	void printDiagnostic(const char choice) const;

protected:
	
	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	scalar* vals;
	
	/// Block-column indices of blocks in data
	index* bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index* browptr;

	/// Number of block-rows
	index nbrows;
	
	/// Stores indices into bcolind if diagonal blocks
	index* diagind;

	/// Storage for factored or inverted diagonal blocks
	Matrix<scalar,Dynamic,bs,RowMajor> dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	Matrix<scalar,Dynamic,bs,RowMajor> iluvals;

	/// Storage for intermediate results in preconditioning operations
	mutable Vector<scalar> ytemp;

	/// Number of sweeps used to build preconditioners
	const short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const short napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
};

/// The limiting case of BSR matrix when block size is 1
template <typename scalar, typename index>
class BSRMatrix<scalar,index,1> : public LinearOperator<scalar, index>
{
public:
	
	/// Minimal initialzation; just sets number of async sweeps	
	BSRMatrix(const short n_buildsweeps, const short n_applysweeps);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of rows
	 * \param[in] bcinds Column indices, simply copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const short nbuildsweeps, const short napplysweeps);
	
	/// A constructor which just wraps a CSR matrix described by its 3 pointers
	/** Does not copy data.
	 */
	BSRMatrix(const index nrows, const index *const brptrs,
		const index *const bcinds, const scalar *const values,
		const short n_buildsweeps, const short n_applysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();

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
	
	/// Insert a block of values into the [matrix](\ref data); not thread-safe
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

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Allocates storage for a vector \ref ytemp required for both SGS and ILU applications
	void allocTempVector();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the number of rows in the matrix
	index dim() const { return nbrows; }
	
	/// Prints out the matrix to a file in dense format
	void printDiagnostic(const char choice) const;

protected:
	
	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	scalar* vals;

	/// Block-column indices of blocks in data
	index* bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index* browptr;

	/// Number of block-rows
	index nbrows;
	
	/// Stores indices into bcolind if diagonal blocks
	index* diagind;

	/// Storage for factored or inverted diagonal blocks
	scalar* dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	scalar* iluvals;
	
	/// Stores scaling vector for async ILU factorization
	scalar* scale;

	/// Storage for intermediate results in preconditioning operations
	mutable scalar* ytemp;

	/// Number of sweeps used to build preconditioners
	const short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const short napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
};

#include "blockmatrices.ipp"

}
#endif
