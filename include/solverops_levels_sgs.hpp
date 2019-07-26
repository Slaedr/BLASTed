/** \file
 * \brief Level-scheduled Gauss-Seidel iterations
 * \author Aditya Kashi
 * \date 2019-02
 */

#ifndef BLASTED_SOLVEROPS_LEVELS_SGS_H
#define BLASTED_SOLVEROPS_LEVELS_SGS_H

#include "solverops_jacobi.hpp"

namespace blasted {

/// Level-scheduled parallel block symmetric Gauss-Seidel iteration
template <typename scalar, typename index, int bs, StorageOptions stor>
class Level_BSGS : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	Level_BSGS(SRMatrixStorage<const scalar, const index>&& matrix);
	~Level_BSGS();

	bool relaxationAvailable() const { return true; }

	/// Compute the preconditioner
	PrecInfo compute();

	/// To apply the preconditioner
	void apply(const scalar *const r, scalar *const __restrict z) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using Preconditioner<scalar,index>::solveparams;
	using SRPreconditioner<scalar,index>::mat;
	using SRPreconditioner<scalar,index>::pmat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;

	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	scalar *ytemp;

	/// Independent levels
	std::vector<index> levels;
};

/// Level-scheduled parallel symmetric Gauss-Seidel iteration
template <typename scalar, typename index>
class Level_SGS : public JacobiSRPreconditioner<scalar,index>
{
public:
	Level_SGS(SRMatrixStorage<const scalar, const index>&& matrix);
	~Level_SGS();

	bool relaxationAvailable() const { return true; }

	/// Compute the preconditioner
	PrecInfo compute();

	/// To apply the preconditioner
	void apply(const scalar *const r, scalar *const __restrict z) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using Preconditioner<scalar,index>::solveparams;
	using SRPreconditioner<scalar,index>::mat;
	using SRPreconditioner<scalar,index>::pmat;
	using JacobiSRPreconditioner<scalar,index>::dblocks;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	scalar *ytemp;

	/// Independent levels
	std::vector<index> levels;
};

}

#endif
