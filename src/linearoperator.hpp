/** \file linearoperator.hpp
 * \brief Abstract interfaces for matrix implementations
 * \author Aditya Kashi
 * \date 2017-07-28
 */

#ifndef __LINEAROPERATOR_H
#define __LINEAROPERATOR_H

/// Contains all of the BLASTed functionality
namespace blasted {

/// Abstract interface for a matrix
/** This is the base class for all matrix implementations in the library.
 */
template <typename scalar, typename index>
class LinearOperator
{
public:
	/// Should compute the matrix vector product of this matrix with one vector
	virtual void apply(const scalar *const x, scalar *const y) const = 0;

	/// Meant to compute needed data for applying the Jacobi preconditioner
	virtual void precJacobiSetup() = 0;
	
	/// Applies any of a class of Jacobi-type preconditioners
	virtual void precJacobiApply(const scalar *const r, scalar *const z) const = 0;

	/// Should compute data needed for Gauss-Seidel type preconditioners
	virtual void precSGSSetup() = 0;

	/// Applies a symmetric Gauss-Seidel type preconditioner
	virtual void precSGSApply(const scalar *const r, scalar *const z) const = 0;

	/// Computes a incomplete lower-upper factorization
	virtual void precILUSetup() = 0;

	/// Applies an LU factorization
	virtual void precILUApply(const scalar *const r, scalar *const z) const = 0;
};

}
#endif
