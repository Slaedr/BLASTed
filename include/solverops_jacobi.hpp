/** \file solverops_jacobi.hpp
 * \brief Header for (block-) Jacobi operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_JACOBI_H
#define BLASTED_SOLVEROPS_JACOBI_H

#include "solverops_base.hpp"

namespace blasted {

/// Block-Jacobi operator for sparse-row matrices
template <typename scalar, typename index, int bs, StorageOptions stopt>
class BJacobiSRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stopt == RowMajor || stopt == ColMajor, "Invalid storage option!");

public:
	BJacobiSRPreconditioner();

	~BJacobiSRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	
	using SRPreconditioner<scalar,index>::mat;
	
	/// Storage for factored or inverted diagonal blocks
	scalar *dblocks;
};

/// Scalar Jacobi operator for sparse-row matrices
/** \note The template parameter for storage order does not matter for this specialization,
 * but it must be specified RowMajor (an arbitrary decision).
 */	
template <typename scalar, typename index>
class JacobiSRPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	JacobiSRPreconditioner();

	~JacobiSRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	
	using SRPreconditioner<scalar,index>::mat;
	
	/// Storage for factored or inverted diagonal blocks
	scalar *dblocks;
};
	
}

#endif