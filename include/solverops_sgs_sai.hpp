/** \file
 * \brief SGS preconditioner applied in parallel by incomplete sparse approximate inverse.
 * \author Aditya Kashi
 */

#ifndef BLASTED_SGS_SAI_H
#define BLASTED_SGS_SAI_H

#include "solverops_base.hpp"

namespace blasted {

/// Block SGS preconditioner applied by (incomplete) sparse approximate inverse
/** The sparsity pattern of the approximate inverse is the same as the original matrix.
 */
template <typename scalar, typename index, int bs, StorageOptions stopt>
class BSGS_SAI : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stopt == RowMajor || stopt == ColMajor, "Invalid storage option!");

public:
	BSGS_SAI(const bool full_spai);

	~BSGS_SAI();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }

	bool relaxationAvailable() const { return false; }
	
	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	
	using Blk = Block_t<scalar,bs,stopt>;
	using Seg = Segment_t<scalar,bs>;
	using SRPreconditioner<scalar,index>::mat;
	using Preconditioner<scalar,index>::solveparams;

	const bool fullsai;
	RawBSRMatrix<scalar,index> sai;
};

}
#endif
