/** \file blockmatrices.hpp
 * \brief Classes for sparse square matrices consisting of square dense blocks of static size
 *
 * There are two kinds of sparse matrix classes. One kind, consisting of \refBSRMatrix,
 * implements a matrix which provides assembling operations and allocates and handles its own data.
 * The other kind, \ref BSRMatrixView, only provides BLAS 2 operations while wrapping an
 * external matrix. It does not allocate or delete storage for the matrix itself.
 *
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef BLOCKMATRICES_H
#define BLOCKMATRICES_H

#include "linearoperator.hpp"
#include <iostream>
#include <limits>
#include <type_traits>
#include <Eigen/Core>

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// A collection of data that represents an immutable compressed sparse block-row square matrix
template <typename scalar, typename index>
struct ConstRawBSRMatrix
{
	const index *const browptr;
	const index *const bcolind;
	const scalar *const vals;
	const index *const diagind;
	const index nbrows;
};

/// A collection of data that represents a compressed sparse block-row square matrix
template <typename scalar, typename index>
struct RawBSRMatrix
{
	index * browptr;
	index * bcolind;
	scalar * vals;
	index * diagind;
	index nbrows;
};

template <typename scalar, typename index>
void destroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat);

template <typename scalar, typename index>
void destroyConstRawBSRMatrix(ConstRawBSRMatrix<scalar,index>& rmat);

/// Block sparse row matrix
/** Dense blocks stored in a (block-) row-major storage order.
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
	BSRMatrix(const int nbuildsweeps, const int napplysweeps);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply (deep) copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply (deep) copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const int nbuildsweeps, const int napplysweeps);
	
	/// A constructor which just wraps a BSR matrix described by 4 arrays
	/** \param[in] n_brows Number of block-rows
	 * \param[in] brptrs Array of block-row pointers
	 * \param[in] bcinds Array of block-column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of diagonal entry pointers
	 * \param[in] n_buildsweeps Number of asynchronous preconditioner build sweeps
	 * \param[in] n_applysweeps Number of asynchronous preconditioner apply sweeps
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrix(const index n_brows, index *const brptrs,
		index *const bcinds, scalar *const values, index *const dinds,
		const int n_buildsweeps, const int n_applysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();

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

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	/** \warning x must not alias z.
	 */
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner. 
	/** Approximately solves D z = r
	 */
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Inverts diagonal blocks and allocates temporary array needed for Gauss-Seidel
	void precSGSSetup();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	/** Approximately solves (D+L) D^(-1) (D+U) z = r
	 * where D, L and U are the diagonal, upper and lower parts of the matrix respectively,
	 * by applying asynchronous Jacobi sweeps.
	 * This block version is adapted from the scalar version in \cite async:anzt_triangular
	 */
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	/** Finds \f$ \tilde{L} \f$ and \f$ \tilde{U} \f$ such that
	 * \f[ \tilde{L} \tilde{U} = A \f]
	 * is approximately satisifed, with \f$ \tilde{L} \f$ unit lower triangular.
	 * The sparsity if A is preserved, so this is ILU(0).
	 * This block version is adapted from the scalar version in \cite ilu:chowpatel . 
	 */
	void precILUSetup();

	/// Applies a block LU factorization L U z = r
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the dimension (number of rows) of the square matrix
	index dim() const { return mat.nbrows*bs; }

protected:

	/** Indicates whether this objects owns data of \ref vals, \ref bcolind, \ref browptr,
	 * \ref diagind
	 */
	bool owner;

	/// The BSR matrix storage
	RawBSRMatrix<scalar,index> mat;

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
	const int nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const int napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
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
	BSRMatrix(const int n_buildsweeps, const int n_applysweeps);

	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of rows
	 * \param[in] bcinds Column indices, simply copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const int nbuildsweeps, const int napplysweeps);
	
	/// A constructor which just wraps a CSR matrix described by 4 arrays
	/** \param[in] nrows Number of rows
	 * \param[in] rptrs Array of row pointers
	 * \param[in] cinds Array of column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of diagonal entry pointers
	 * \param[in] n_buildsweeps Number of asynchronous preconditioner build sweeps
	 * \param[in] n_applysweeps Number of asynchronous preconditioner apply sweeps
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrix(const index nrows, index *const rptrs,
		index *const cinds, scalar *const values, index *const dinds,
		const int n_buildsweeps, const int n_applysweeps);

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

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for the Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Inverts diagonal blocks and allocates a temporary array needed for Gauss-Seidel
	void precSGSSetup();

	/// Applies a block symmetric Gauss-Seidel preconditioner
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the number of rows in the matrix
	index dim() const { return mat.nbrows; }
	
protected:
	
	/** Indicates whether the data (\ref vals, \ref bcolind, \ref browptr, \ref diagind)
	 * is owned by this object
	 */
	bool owner;
	
	/// The CSR matrix data	
	RawBSRMatrix<scalar,index> mat;

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
	const int nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const int napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
};

/// A BSR matrix that is formed by wrapping a pre-existing read-only matrix
/** StorageOptions is an Eigen type describing storage options, which we use here to
 * specify whether storage within blocks is row-major or column-major. If each block is row-major,
 * stopt should be Eigen::RowMajor else it shoule be Eigen::ColMajor.
 * The blocks are always arranged block-row-wise relative to each other.
 */
template <typename scalar, typename index, int bs, StorageOptions stopt>
class BSRMatrixView : public MatrixView<scalar, index>
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
	 * \param[in] n_buildsweeps Number of asynchronous preconditioner build sweeps
	 * \param[in] n_applysweeps Number of asynchronous preconditioner apply sweeps
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	BSRMatrixView(const index n_brows, const index *const brptrs,
		const index *const bcinds, const scalar *const values, const index *const dinds,
		const int n_buildsweeps, const int n_applysweeps);

	/// Cleans up temporary data needed for preconditioning operations
	virtual ~BSRMatrixView();

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	/** \warning x must not alias z.
	 */
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner. 
	/** Approximately solves D z = r
	 */
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Inverts diagonal blocks and allocates temporary array needed for Gauss-Seidel
	void precSGSSetup();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	/** Approximately solves (D+L) D^(-1) (D+U) z = r
	 * where D, L and U are the diagonal, upper and lower parts of the matrix respectively,
	 * by applying asynchronous Jacobi sweeps.
	 * This block version is adapted from the scalar version in \cite async:anzt_triangular
	 */
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	/** Finds \f$ \tilde{L} \f$ and \f$ \tilde{U} \f$ such that
	 * \f[ \tilde{L} \tilde{U} = A \f]
	 * is approximately satisifed, with \f$ \tilde{L} \f$ unit lower triangular.
	 * The sparsity if A is preserved, so this is ILU(0).
	 * This block version is adapted from the scalar version in \cite ilu:chowpatel . 
	 */
	void precILUSetup();

	/// Applies a block LU factorization L U z = r
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the dimension (number of rows) of the square matrix
	index dim() const { return mat.nbrows*bs; }

protected:

	/// The BSR matrix wrapper
	ConstRawBSRMatrix<scalar,index> mat;

	/// Storage for factored or inverted diagonal blocks
	scalar *dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	scalar *iluvals;

	/// Storage for intermediate results in preconditioning operations
	mutable Vector<scalar> ytemp;

	/// Number of sweeps used to build preconditioners
	const int nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const int napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
};

/// A CSR matrix formed by wrapping a read-only matrix
/**
 * On destruct, cleans up only its own data that are needed for preconditioning operations.
 */
template <typename scalar, typename index>
class CSRMatrixView : public MatrixView<scalar, index>
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
	 * \param[in] n_buildsweeps Number of asynchronous preconditioner build sweeps
	 * \param[in] n_applysweeps Number of asynchronous preconditioner apply sweeps
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	CSRMatrixView(const index nrows, const index *const rptrs,
		const index *const cinds, const scalar *const values, const index *const dinds,
		const int n_buildsweeps, const int n_applysweeps);

	/// De-allocates temporary storage only, not the matrix storage itself
	virtual ~CSRMatrixView();

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for the Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Inverts diagonal blocks and allocates a temporary array needed for Gauss-Seidel
	void precSGSSetup();

	/// Applies a block symmetric Gauss-Seidel preconditioner
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
	
	/// Returns the number of rows in the matrix
	index dim() const { return mat.nbrows; }

protected:
	
	/// The CSR matrix data	
	ConstRawBSRMatrix<scalar,index> mat;

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
	const int nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const int napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;
};

}
#endif
