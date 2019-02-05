/** \file relaxation_chaotic.hpp
 * \brief Asynchronous (chaotic) relaxation
 * \author Aditya Kashi
 * \date 2018-05
 */

#ifndef BLASTED_RELAXATION_CHAOTIC
#define BLASTED_RELAXATION_CHAOTIC

#include "solverops_jacobi.hpp"

namespace blasted {

/// Point-block version of the Chazan-Miranker chaotic relaxation \cite async:chazan_1969
/** Does not use `-blasted_async_sweeps'.
 */
template<typename scalar, typename index, int bs, StorageOptions stor>
class ChaoticBlockRelaxation : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	/// Constructor
	/** \param napplysweeps In case of preconditioning, number of sweeps to use per preconditioner
	 *    application. Ignored for relaxation.
	 * \param threadchunksize Number of iterations to assign to a thread at a time
	 */
	ChaoticBlockRelaxation(const int napplysweeps, const int threadchunksize);

	/// Async. forward block-Gauss-Seidel preconditioner
	void apply(const scalar *const b, scalar *const __restrict x) const;

	/// Carry out chaotic block relaxation
	/** For this solver, tolerance checking is never done irrespective of
	 * \ref Precontitioner::setApplyParams.
	 * Note that no initial condition is set here - the prior contents of x are used as
	 * the initial guess.
	 * \param b The right hand side in Ax=b
	 * \param x The solution vector, initially containing the initial guess
	 */
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	using Preconditioner<scalar,index>::solveparams;

	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	const int napplysweeps;
	const int thread_chunk_size;
};

/// Chazan-Miranker chaotic relaxation \cite async:chazan_1969
/** Does not use `-blasted_async_sweeps'.
 */
template<typename scalar, typename index>
class ChaoticRelaxation : public JacobiSRPreconditioner<scalar,index>
{
public:
	/// Constructor
	/** \param n_applysweeps In case of preconditioning, number of sweeps to use per preconditioner
	 *    application. Ignored for relaxation.
	 * \param threadchunksize Number of iterations to assign to a thread at a time
	 */
	ChaoticRelaxation(const int n_applysweeps, const int threadchunksize);

	/// Async. forward Gauss-Seidel preconditioner
	void apply(const scalar *const b, scalar *const __restrict x) const;

	/// Carry out chaotic relaxation
	/** For this solver, tolerance checking is never done irrespective of
	 * \ref Precontitioner::setApplyParams.
	 * Note that no initial condition is set here - the prior contents of x are used as
	 * the initial guess.
	 * \param b The right hand side in Ax=b
	 * \param x The solution vector, initially containing the initial guess
	 */
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using JacobiSRPreconditioner<scalar,index>::dblocks;
	using Preconditioner<scalar,index>::solveparams;

	const int napplysweeps;
	const int thread_chunk_size;
};

}

#endif
