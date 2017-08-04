/** \file linearoperator.hpp
 * \brief Abstract interfaces for matrix implementations
 * \author Aditya Kashi
 * \date 2017-07-28
 */

#ifndef __LINEAROPERATOR_H
#define __LINEAROPERATOR_H

#include <Eigen/Core>

#ifdef DEBUG
#define __restrict__ 
#define __restrict
#endif

/// Contains all of the BLASTed functionality
namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;

template<typename scalar>
using Vector = Eigen::Matrix<scalar,Dynamic,1>;

/// Abstract interface for a matrix
/** This is the base class for all matrix implementations in the library.
 */
template <typename scalar, typename index>
class LinearOperator
{
public:
	/// To insert a (contiguous) block of values into the matrix
	virtual void submitBlock(const index starti, const index startj, 
			const index bsizei, const index bsizej, const scalar *const buffer) = 0;

	/// Should compute the matrix vector product of this matrix with one vector,
	/// scaled by a constant
	virtual void apply(const scalar a, const scalar *const x, 
			scalar *const __restrict__ y) const = 0;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict__ x, 
			const scalar b, const scalar *const y,
			scalar *const z) const = 0;

	/// Meant to compute needed data for applying the Jacobi preconditioner
	virtual void precJacobiSetup() = 0;
	
	/// Applies any of a class of Jacobi-type preconditioners
	virtual void precJacobiApply(const scalar *const r, scalar *const __restrict__ z) const = 0;

	/// Should compute data needed for Gauss-Seidel type preconditioners
	virtual void precSGSSetup() = 0;

	/// Applies a symmetric Gauss-Seidel type preconditioner
	virtual void precSGSApply(const scalar *const r, scalar *const __restrict__ z) const = 0;

	/// Computes a incomplete lower-upper factorization
	virtual void precILUSetup() = 0;

	/// Applies an LU factorization
	virtual void precILUApply(const scalar *const r, scalar *const __restrict__ z) const = 0;
};

}
#endif
