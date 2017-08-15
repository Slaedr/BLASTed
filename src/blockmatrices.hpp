/** \file blockmatrices.hpp
 * \brief Classes for sparse matrices consisting of dense blocks
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef __BLOCKMATRICES_H
#define __BLOCKMATRICES_H

// get around dependent templates for Eigen
#define SEG template segment
#define BLK template block

#include "linearoperator.hpp"
#include <iostream>
#include <Eigen/LU>

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// Block sparse row matrix
/** Dense blocks stored in a (block-) row-major storage order.
 * \warning Do NOT use unsigned types as index; that will cause SGS and ILU preconditioners to fail.
 */
template <typename scalar, typename index, int bs>
class BSRMatrix : public LinearOperator<scalar, index>
{
protected:
	
	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	scalar* vals;

	/// Has space for non-zero values \ref vals been allocated by this class?
	const bool isAllocVals;
	
	/// Block-column indices of blocks in data
	index* bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index* browptr;

	/// Number of block-rows
	const index nbrows;
	
	/// Stores indices into bcolind if diagonal blocks
	index* diagind;

	/// Storage for factored or inverted diagonal blocks
	Matrix<scalar,Dynamic,bs,RowMajor> dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	Matrix<scalar,Dynamic,bs,RowMajor> iludata;

	/// Storage for intermediate results in preconditioning operations
	mutable Vector<scalar> ytemp;

	/// Number of sweeps used to build preconditioners
	const short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const short napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;

public:
	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const short nbuildsweeps, const short napplysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();

	/// Sets all stored entries of the matrix to zero
	void setAllZero();

	/// Sets diagonal blocks to zero
	void setDiagZero();
	
	/// Insert a block of values into the [matrix](\ref data); not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location of the matrix at the same time.
	 */
	void submitBlock(const index starti, const index startj, 
			const scalar *const buffer, const long param1, const long param2);

	/// Update a (contiguous) block of values into the [matrix](\ref vals)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 */
	void updateBlock(const index starti, const index startj, 
			const scalar *const buffer, const long param1, const long param2);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The block-row whose diagonal block is to be updated
	 * \param[in] buffer The values making up the block to be added
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer, const long param);

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
	
	index dim() const { return nbrows*bs; }
};

/// The limiting case of BSR matrix when block size is 1
template <typename scalar, typename index>
class BSRMatrix<scalar,index,1> : public LinearOperator<scalar, index>
{
protected:
	
	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	scalar* vals;

	/// Has space for non-zero values \ref vals been allocated by this class?
	const bool isAllocVals;

	/// Number of block-rows
	const index nbrows;
	
	/// Stores indices into bcolind if diagonal blocks
	index* diagind;

	/// Storage for factored or inverted diagonal blocks
	scalar* dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	scalar* iludata;
	
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

public:
	
	/// Block-column indices of blocks in data
	index* bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index* browptr;
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
			const scalar *const buffer, const long bsi, const long bsj);

	/// Update a (contiguous) block of values into the [matrix](\ref vals)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 * \param[in] starti The (global) row index of the first row of the block to be inserted
	 * \param[in] startj The (global) column index of the first row of the block to be inserted
	 * \param[in] bsi The number of rows in the block being inserted
	 * \param[in] bsj The number of columns in the block being inserted
	 */
	void updateBlock(const index starti, const index startj, 
			const scalar *const buffer, const long bsi, const long bsj);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The row whose diagonal block is to be updated
	 * \param[in] buffer The values making up the block to be added
	 * \param[in] bs Size of the square block to be updated
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer, const long bs);

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
	
	index dim() const { return nbrows; }
};

#include "blockmatrices.ipp"

}
#endif
