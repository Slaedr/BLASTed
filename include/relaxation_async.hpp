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
/** Does not use `-blasted_async_sweeps'.
 */
template<typename scalar, typename index, int bs, StorageOptions stor>
class ChaoticBlockRelaxation : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	ChaoticBlockRelaxation();

	/// Carry out chaotic block relaxation
	/** For this solver, tolerance checking is never done irrespective of
	 * \ref Precontitioner::setApplyParams.
	 * Note that no initial condition is set here - the prior contents of x are used as
	 * the initial guess.
	 * \param b The right hand side in Ax=b
	 * \param x The solution vector, initially containing the initial guess
	 */
	void apply(const scalar *const b, scalar *const __restrict x) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	using Preconditioner<scalar,index>::solveparams;

	const int thread_chunk_size;
};

/// Asynchronous block SGS relaxation
/** Fully asynchronous relaxation in which every "sweep" is followed by a sweep in the reverse
 * direction. Unlike the ABSGS preconditioner, this operation does not use `-blasted_async_sweeps';
 * this is fully asychronous relaxation.
 */
template<typename scalar, typename index, int bs, StorageOptions stor>
class ABSGS_Relaxation : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	ABSGS_Relaxation();

	/// Carry out chaotic block SGS relaxation
	/** For this solver, tolerance checking is never done irrespective of
	 * \ref Precontitioner::setApplyParams.
	 * \param b The right hand side in Ax=b
	 * \param x The solution vector, initially containing the initial guess
	 */
	void apply(const scalar *const b, scalar *const __restrict x) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	using Preconditioner<scalar,index>::solveparams;

	const int thread_chunk_size;
};

}

#endif
