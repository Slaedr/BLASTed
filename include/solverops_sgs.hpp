/** \file solverops_sgs.hpp
 * \brief Header for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_SGS_H
#define BLASTED_SOLVEROPS_SGS_H

#include "solverops_jacobi.hpp"

namespace blasted {

/// Asynchronous block-SGS operator for sparse-row matrices
/** \warning While re-wrapping a different matrix, make sure it's of the same dimension
 * as the previous one.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
class ABSGS_SRPreconditioner : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	/// Create block SGS preconditioner
	/** \param napplysweeps Number of asynchronous application sweeps
	 */
	ABSGS_SRPreconditioner(const int napplysweeps);

	~ABSGS_SRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }

	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;

	const int napplysweeps;
	const int thread_chunk_size;
};

/// Asynchronous scalar SGS operator for sparse-row matrices
template <typename scalar, typename index>
class ASGS_SRPreconditioner : public JacobiSRPreconditioner<scalar,index>
{
public:
	/// Create asynchronous scalar SGS preconditioner
	/** \param napplysweeps Number of asynchronous application sweeps
	 */
	ASGS_SRPreconditioner(const int napplysweeps);

	~ASGS_SRPreconditioner();

	/// Returns the number of rows
	index dim() const { return mat.nbrows; }

	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using JacobiSRPreconditioner<scalar,index>::dblocks;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;

	const int napplysweeps;
	const int thread_chunk_size;
};

}

#endif
