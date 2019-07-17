/** \file
 * \brief Sparse approximate inverse preconditioner
 */

#ifndef BLASTED_SOLVEROPS_SAI_H
#define BLASTED_SOLVEROPS_SAI_H

#include "solverops_base.hpp"

namespace blasted {

/// Data needed for SAI computations
template <typename index>
class LeftSAIImpl;

/// 
template <typename scalar, typename index, int bs, StorageOptions stor>
class LeftSAIPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	LeftSAIPreconditioner();

	bool relaxationAvailable() const { return false; }

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// To apply relaxation - not available; throws an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	/// SAI preprocessing data
	LeftSAIImpl<index> impl;
};

}

#endif
