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
template <typename scalar, typename index, unsigned int bsize>
class BSRMatrix : public LinearOperator<scalar, index>
{
protected:
	/// Entries
	scalar * data;
	
	/// Block-column indices of blocks in data
	index * bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index * browptr;
	
	/// Stores indices into bcolind if diagonal blocks
	index * diagblocks;

public:
	/// Allocates space for the matrix based on the number of blocks needed for each block-row
	BSRMatrix(const index nbrows, const index *const blocksperrow);

	/// De-allocates memory
	virtual ~BSRMatrix();
	
	/// Insert a (contiguous) block of values into the matrix
	void submitBlock(const index starti, const index startj, 
			const index bsizei, const index bsizej, const scalar *const buffer);

	/// Computes the matrix vector product of this matrix with one vector
	void apply(const scalar *const x, scalar *const y) const;

	/// Computes inverse or dense factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const z) const;

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	void precSGSApply(const scalar *const r, scalar *const z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const z) const;
};

}
#endif
