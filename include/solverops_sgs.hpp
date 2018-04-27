/** \file solverops_sgs.hpp
 * \brief Header for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_SGS_H
#define BLASTED_SOLVEROPS_SGS_H

#include "solverops_jacobi.hpp"

namespace blasted {

/// Asynchronous block-SGS operator for sparse-row matrices
template <typename scalar, typename index, int bs, StorageOptions stor>
class ASGS_SRPreconditioner : public JacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	ASGS_SRPreconditioner();

	~ASGS_SRPreconditioner();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;
};

/// Asynchronous scalar SGS operator for sparse-row matrices
template <typename scalar, typename index, StorageOptions stor>
class ASGS_SRPreconditioner<scalar,index,1,stor>
	: public JacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	ASGS_SRPreconditioner();

	~ASGS_SRPreconditioner();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;
};

}

#endif
