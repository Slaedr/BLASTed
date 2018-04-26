/** \file solverops_ilu0.hpp
 * \brief Header for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_ILU0_H
#define BLASTED_SOLVEROPS_ILU0_H

#include "solverops_base.hpp"

namespace blasted {

/// Asynchronous (block-) ILU(0) operator for sparse-row matrices
template <typename scalar, typename index, int bs, StorageOptions stor>
class AILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stopt == RowMajor || stopt == ColMajor, "Invalid storage option!");

public:
	/** \param serialfactor If true, the preconditioner is computed sequentially
	 * \param serialapply If true, the preconditioner is applied sequentially
	 */
	AILU0_SRPreconditioner(const bool serialfactor=false, const bool serialapply=false);

	~AILU0_SRPreconditioner();
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// Storage for L and U factors
	scalar *iluvals;
};

} // end namespace

#endif
