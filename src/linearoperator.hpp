/** \file linearoperator.hpp
 * \brief Abstract interfaces for matrix implementations
 * \author Aditya Kashi
 * \date 2017-07-28
 */

#ifndef LINEAROPERATOR_H
#define LINEAROPERATOR_H

#ifdef DEBUG
#define __restrict__ 
#define __restrict
#endif

/// Contains all of the BLASTed functionality
namespace blasted {

/// Abstract interface for a matrix
/** This is the base class for all matrix implementations in the library.
 */
template <typename scalar, typename index>
class LinearOperator
{
protected:
	/// Encodes some indication of what kind of storage is used
	char _type;

public:
	LinearOperator(const char storagetype) : _type{storagetype}
	{ }

	virtual ~LinearOperator() { }

	char type() { return _type; }

	/// Sets the structure of the matrix using supplied vectors
	/** See the documentation for the subclasses for requirements on the inputs.
	 */
	virtual void setStructure(const index n, 
			const index *const vec1, const index *const vec2) = 0;

	/// Sets all non-zero entries to explicitly stored zeros
	virtual void setAllZero() = 0;

	/// Sets diagonals or diagonal blocks to zero, depending on the type of sorage
	virtual void setDiagZero() = 0;

	/// To insert a (contiguous) block of values into the matrix
	/**
	 * \param[in] starti The row index at which the block starts;
	 *              Note that this must not be the block-row index for any matrix type.
	 * \param[in] startj The row index at which the block starts;
	 *              Note that this must not be the block-column index for any matrix type.
	 */
	virtual void submitBlock(const index starti, const index startj, 
		const scalar *const buffer, const index param1, const index param2) = 0;
	
	/// Supposed to add to a contiguous block of the matrix in a thread-safe manner
	/**
	 * \param[in] starti The row index at which the block starts;
	 *              Note that this must not be the block-row index for any matrix type.
	 * \param[in] startj The row index at which the block starts;
	 *              Note that this must not be the block-column index for any matrix type.
	 */
	virtual void updateBlock(const index starti, const index startj, 
		const scalar *const buffer, const index param1, const index param2) = 0;
	
	/// Supposed to update diagonal entries for point matrices 
	/// and diagonal blocks for block matrices
	/**
	 * \param[in] starti The row index at which the block starts;
	 *              Note that this must not be the block-row index for any matrix type.
	 * \param[in] Any parameter needed by implementations
	 */
	virtual void updateDiagBlock(const index starti, const scalar *const buffer,
			const index param1) = 0;

	/// Scales all entries of the matrix by scalar
	virtual void scaleAll(const scalar factor) = 0;

	/// Should compute the matrix vector product of this matrix with one vector,
	/// scaled by a constant
	virtual void apply(const scalar a, const scalar *const x, 
			scalar *const __restrict y) const = 0;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const = 0;

	/// Meant to compute needed data for applying the Jacobi preconditioner
	virtual void precJacobiSetup() = 0;
	
	/// Applies any of a class of Jacobi-type preconditioners
	virtual void precJacobiApply(const scalar *const r, scalar *const __restrict z) const = 0;

	/// Applies a symmetric Gauss-Seidel type preconditioner
	virtual void precSGSApply(const scalar *const r, scalar *const __restrict z) const = 0;

	/// Computes a incomplete lower-upper factorization
	virtual void precILUSetup() = 0;

	/// Applies an LU factorization
	virtual void precILUApply(const scalar *const r, scalar *const __restrict__ z) const = 0;

	/// Allocates storage for a vector required for both SGS and ILU applications
	virtual void allocTempVector() = 0;
	
	/// Returns the dimension (number of rows) of the operator
	virtual index dim() const = 0;

	/// Print parts of the matrix for diagnostics
	virtual void printDiagnostic(const char choice) const { }
};

}
#endif
