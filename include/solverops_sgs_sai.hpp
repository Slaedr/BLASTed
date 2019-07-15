/** \file
 * \brief SGS preconditioner applied in parallel by incomplete sparse approximate inverse.
 * \author Aditya Kashi
 */

#ifndef BLASTED_SGS_SAI_H
#define BLASTED_SGS_SAI_H

#include "solverops_base.hpp"
#include "sai.hpp"

namespace blasted {

/// Block SGS preconditioner applied by (incomplete) sparse approximate inverse
/** The sparsity pattern of the approximate inverse is the same as the original matrix.
 * We use the left approximate inverse as it is easy to implement for CSR-stored matrices. That is,
 * we compute \f$ \min_{m_k} \lVert m_k^T A - e_k^T \rVert_2 = \min_{m_k} \lVert A^T m_k - e_k \rVert_2 \f$
 * for each column k of the approximate inverse M.
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

	LeftSAIPattern<index> saipattern;

	RawBSRMatrix<scalar,index> saiL;
	RawBSRMatrix<scalar,index> saiU;

	using LMatrix = Matrix<scalar,Dynamic,Dynamic,ColMajor>;

	/// Sets up data structures the first time
	/** Computes sizes of lower and upper approximate inverses and allocates storage.
	 * Also calls \ref compute_pattern
	 */
	void initialize();

	/// Computes the pattern of each local least-squares problem
	void compute_pattern();

	__attribute__((always_inline))
	void compute_SAI_lower(const index col);
	__attribute__((always_inline))
	void compute_SAI_upper(const index col);
	// __attribute__((always_inline))
	// LMatrix compute_LHS_SAI_upper(const index col) const;
	// __attribute__((always_inline))
	// LMatrix compute_LHS_SAI_lower(const index col) const;
	// __attribute__((always_inline))
	// LMatrix compute_LHS_incompSAI_upper(const index col) const;
	// __attribute__((always_inline))
	// LMatrix compute_LHS_incompSAI_lower(const index col) const;
};

}
#endif
