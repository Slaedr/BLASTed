/** \file solverops_ilu0.hpp
 * \brief Header for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_ILU0_H
#define BLASTED_SOLVEROPS_ILU0_H

#include "solverops_base.hpp"

namespace blasted {

/// Asynchronous block-ILU(0) preconditioner for sparse-row matrices
/** Finds \f$ \tilde{L} \f$ and \f$ \tilde{U} \f$ such that
 * \f[ \tilde{L} \tilde{U} = A \f]
 * is approximately satisifed, with \f$ \tilde{L} \f$ unit lower triangular.
 * The sparsity if A is preserved, so this is ILU(0).
 * This block version is adapted from the scalar version in \cite ilu:chowpatel . 
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
class AsyncBlockILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");

public:
	/** \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AsyncBlockILU0_SRPreconditioner(const int nbuildsweeps, const int napplysweeps,
	                       const bool threadedfactor=true, const bool threadedapply=true);

	~AsyncBlockILU0_SRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }
	
	/// Compute the preconditioner
	void compute();

	/// Applies a block LU factorization L U z = r
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	
	/// Storage for L and U factors
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	scalar *iluvals;

	/// Matrix used to scale the original matrix before factorization
	scalar *scale;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool threadedfactor;                      ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;                       ///< True for thread-parallel LU application
	const int nbuildsweeps;
	const int napplysweeps;
	const int thread_chunk_size;
};

/// Asynchronous scalar ILU(0) operator for sparse-row matrices
template <typename scalar, typename index>
class AsyncILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	/** \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AsyncILU0_SRPreconditioner(const int nbuildsweeps, const int napplysweeps,
	                       const bool threadedfactor=true, const bool threadedapply=true);

	~AsyncILU0_SRPreconditioner();

	/// Returns the number of rows
	index dim() const { return mat.nbrows; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	
	/// Storage for L and U factors
	scalar *iluvals;

	/// Matrix used to scale the original matrix before factorization
	scalar *scale;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool threadedfactor;                      ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;                       ///< True for thread-parallel LU application
	const int nbuildsweeps;
	const int napplysweeps;
	const int thread_chunk_size;
};

} // end namespace

#endif
