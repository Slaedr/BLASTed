/** \file blockmatrices.hpp
 * \brief Classes for sparse matrices consisting of dense blocks
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef __BLOCKMATRICES_H
#define __BLOCKMATRICES_H

#include "linearoperator.hpp"

namespace blasted {

/// Block sparse row matrix
/** Dense blocks stored in a (block-) row-major storage order.
 */
template <typename scalar, typename index, size_t bsize>
class BSRMatrix : public LinearOperator<scalar, index>
{
protected:
	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	Matrix<scalar,Dynamic,bs,RowMajor> data;
	
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
	Vector<scalar> ytemp;

	/// Number of sweeps used to build preconditioners
	const unsigned short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const unsigned short napplyweeps;

	/// Thread chunk size for OpenMP parallelism
	const unsigned int thread_chunk_size;

public:
	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const unsigned short nbuildsweeps, const unsigned short napplysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();
	
	/// Insert a block of values into the [matrix](\ref data); not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location in \ref data at the same time.
	 */
	void submitBlock(const index starti, const index startj, 
			const scalar *const buffer, const size_t param1, const size_t param2);

	/// Update a (contiguous) block of values into the [matrix](\ref data)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 */
	void updateBlock(const index starti, const index startj, 
			const scalar *const buffer, const size_t param1, const size_t param2);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The block-row whose diagonal block is to be updated
	 * \param[in] buffer The values making up the block to be added
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer);

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __retrict__ y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict__ x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const __restrict__ z) const;

	/// Allocates storage for a vector \ref ytemp required for both SGS and ILU applications
	void allocTempVector();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	void precSGSApply(const scalar *const r, scalar *const __restrict__ z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const __restrict__ z) const;
};

}
#endif
