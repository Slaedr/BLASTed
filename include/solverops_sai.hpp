/** \file
 * \brief Sparse approximate inverse preconditioner
 */

#ifndef BLASTED_SOLVEROPS_SAI_H
#define BLASTED_SOLVEROPS_SAI_H

#include "solverops_base.hpp"

namespace blasted {

/// Data needed for SAI computations
template <typename scalar, typename index>
struct LeftSAIImpl;

/// A block SAI(1) preconditioner
/** Left sparse approximate inverse with static sparsity pattern corresponding to the sparsity
 * pattern of the original matrix.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
class LeftSAIPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");
public:
	LeftSAIPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
	                      const int thread_chunk_size);

	bool relaxationAvailable() const { return false; }

	/// Compute the approximate inverse
	PrecInfo compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// To apply relaxation - not available; throws an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::pmat;

	/// Size of each chunk of work-items
	const int thread_chunk_size;

	/// SAI preprocessing data
	LeftSAIImpl<scalar,index> impl;
};

}

#endif
