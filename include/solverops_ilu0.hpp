/** \file solverops_ilu0.hpp
 * \brief Header for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_ILU0_H
#define BLASTED_SOLVEROPS_ILU0_H

#include "solverops_base.hpp"

namespace blasted {

/// Asynchronous block-ILU(0) operator for sparse-row matrices
template <typename scalar, typename index, int bs, StorageOptions stor>
class AILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stopt == RowMajor || stopt == ColMajor, "Invalid storage option!");

public:
	/** \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AILU0_SRPreconditioner(const bool threadedfactor=true, const bool threadedapply=true);

	~AILU0_SRPreconditioner();
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// Storage for L and U factors
	scalar *iluvals;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool threadedfactor;                      ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;                       ///< True for thread-parallel LU application
};

/// Asynchronous scalar ILU(0) operator for sparse-row matrices
template <typename scalar, typename index, StorageOptions stor>
class AILU0_SRPreconditioner<scalar,index,1,stor> : public SRPreconditioner<scalar,index>
{
public:
	/** \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AILU0_SRPreconditioner(const bool threadedfactor=true, const bool threadedapply=true);

	~AILU0_SRPreconditioner();
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// Storage for L and U factors
	scalar *iluvals;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool threadedfactor;                      ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;                       ///< True for thread-parallel LU application
};

} // end namespace

#endif
