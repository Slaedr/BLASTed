/** \file solverops_ilu0.hpp
 * \brief Header for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_ILU0_H
#define BLASTED_SOLVEROPS_ILU0_H

#include "solverops_base.hpp"

namespace blasted {

/// Asynchronous block-ILU(0) preconditioner for sparse-row matrices
template <typename scalar, typename index, int bs, StorageOptions stor>
class ABILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");

public:
	/** \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	ABILU0_SRPreconditioner(const int nbuildsweeps, const int napplysweeps,
	                       const bool threadedfactor=true, const bool threadedapply=true);

	~ABILU0_SRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }
	
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

/// Asynchronous scalar ILU(0) operator for sparse-row matrices
template <typename scalar, typename index>
class AILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	/** \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AILU0_SRPreconditioner(const int nbuildsweeps, const int napplysweeps,
	                       const bool threadedfactor=true, const bool threadedapply=true);

	~AILU0_SRPreconditioner();

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
