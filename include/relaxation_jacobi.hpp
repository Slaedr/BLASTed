/** \file relaxation_jacobi.hpp
 * \brief Jacobi relaxation
 * \author Aditya Kashi
 * \date 2018-04
 */

#ifndef BLASTED_RELAXATION_JACOBI
#define BLASTED_RELAXATION_JACOBI

#include "solverops_jacobi.hpp"

namespace blasted {

template<typename scalar, typename index, int bs, StorageOptions stor>
class BJacobiRelaxation : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	//BJacobiRelaxation();

	/// Carry out block-Jacobi relaxation
	/** Note that no initial condition is set here - the prior contents of x are used as
	 * the initial guess.
	 */
	void apply(const scalar *const b, scalar *const __restrict x) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;
	using Preconditioner<scalar,index>::solveparams;
};

}

#endif
