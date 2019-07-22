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
	BJacobiSRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix);

	~BJacobiSRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }

	bool relaxationAvailable() const { return true; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	
	using SRPreconditioner<scalar,index>::mat;
	using Preconditioner<scalar,index>::solveparams;
	using Blk = Block_t<scalar,bs,stopt>;
	using Seg = Segment_t<scalar,bs>;
	
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
	JacobiSRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix);

	~JacobiSRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows; }

	bool relaxationAvailable() const { return true; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	
	using SRPreconditioner<scalar,index>::mat;
	using Preconditioner<scalar,index>::solveparams;
	
	/// Storage for factored or inverted diagonal blocks
	scalar *dblocks;
};
	
}

#endif
