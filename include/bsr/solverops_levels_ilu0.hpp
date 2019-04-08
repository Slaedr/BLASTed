/** \file
 * \brief Iterations which use level-scheduling for either factorization or application or both
 * \author Aditya Kashi
 */

#ifndef BLASTED_LEVEL_ILU0_H
#define BLASTED_LEVEL_ILU0_H

#include "solverops_ilu0.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
class Async_Level_BlockILU0 : public AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
{
public:
	Async_Level_BlockILU0(const int nbuildsweeps, const int thread_chunk_size,
	                      const FactInit fact_inittype, const bool threadedfactor=true,
	                      const bool compute_remainder = false);

	~Async_Level_BlockILU0();

	/// Calls the asynchronous factorization routine. The first invocation also computes levels.
	void compute();

	/// Applies a block LU factorization L U z = r
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::plist;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::iluvals;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::scale;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::ytemp;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::nbuildsweeps;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::thread_chunk_size;
	using AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::threadedfactor;

	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	std::vector<index> levels;
};

template <typename scalar, typename index>
class Async_Level_ILU0 : public AsyncILU0_SRPreconditioner<scalar,index>
{
public:
	Async_Level_ILU0(const int nbuildsweeps, const int thread_chunk_size,
	                      const FactInit fact_inittype, const bool threadedfactor=true,
	                      const bool compute_remainder = false);

	~Async_Level_ILU0();

	/// Calls the asynchronous factorization routine. The first invocation also computes levels.
	void compute();

	/// Applies a block LU factorization L U z = r
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using AsyncILU0_SRPreconditioner<scalar,index>::plist;
	using AsyncILU0_SRPreconditioner<scalar,index>::iluvals;
	using AsyncILU0_SRPreconditioner<scalar,index>::scale;
	using AsyncILU0_SRPreconditioner<scalar,index>::ytemp;
	using AsyncILU0_SRPreconditioner<scalar,index>::nbuildsweeps;
	using AsyncILU0_SRPreconditioner<scalar,index>::thread_chunk_size;
	using AsyncILU0_SRPreconditioner<scalar,index>::threadedfactor;

	std::vector<index> levels;
};

}

#endif
