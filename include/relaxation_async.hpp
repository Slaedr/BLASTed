/** \file relaxation_async.hpp
 * \brief Asynchronous (chaotic) relaxation
 * \author Aditya Kashi
 * \date 2018-05
 */

#ifndef BLASTED_RELAXATION_ASYNC
#define BLASTED_RELAXATION_ASYNC

#include "solverops_jacobi.hpp"

namespace blasted {

/// Point-block version of the Chazan-Miranker chaotic relaxation 
template<typename scalar, typename index, int bs, StorageOptions stor>
class ChaoticBlockRelaxation : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	ChaoticBlockRelaxation();

	/// Carry out chaotic block relaxation
	/** Note that no initial condition is set here - the prior contents of x are used as
	 * the initial guess.
	 */
	void apply(const scalar *const b, scalar *const __restrict x) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	using Preconditioner<scalar,index>::solveparams;

	const int thead_chunk_size;
};

}

#endif
