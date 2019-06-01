/** \file
 * \brief SGS preconditioner applied in parallel by incomplete sparse approximate inverse.
 * \author Aditya Kashi
 */

#ifndef BLASTED_SGS_SAI_H
#define BLASTED_SGS_SAI_H

#include "solverops_base.hpp"
#include "adjacency.hpp"

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

	RawBSRMatrix<scalar,index> saiL;
	RawBSRMatrix<scalar,index> saiU;

	using LMatrix = Matrix<scalar,Dynamic,Dynamic,ColMajor>;
	using IArray = Eigen::Array<index,Dynamic,Dynamic,RowMajor>;

	/// Indexing required for assembling the local least-squares problem and scattering the result
	/** Below, Minv refers to the (incomplete) approximate inverse matrix
	 */
	struct LSIndices
	{
		/// For each column of Minv, for each entry, the location in Minv
		std::vector<index> midx;
		/// Pointers into \ref midx for each column of Minv
		std::vector<index> mptr;

		/// For each column of Minv, locations (in the original matrix) of each entry of the local LHS
		std::vector<index> ls_loc;
		/// Pointers into \ref ls_loc for each column of Minv
		std::vector<index> ijptr;

		/// Location of identity block in the RHS vector of each column's local problem
		std::vector<int> idtloc;
	} lsindices;

	/// Sets up data structures the first time
	/** Computes sizes of lower and upper approximate inverses and allocates storage.
	 */
	void initialize();

	__attribute__((always_inline))
	LMatrix compute_LHS_SAI_upper(const index col) const;
	__attribute__((always_inline))
	LMatrix compute_LHS_SAI_lower(const index col) const;
	__attribute__((always_inline))
	LMatrix compute_LHS_incompSAI_upper(const index col) const;
	__attribute__((always_inline))
	LMatrix compute_LHS_incompSAI_lower(const index col) const;
};

}
#endif
