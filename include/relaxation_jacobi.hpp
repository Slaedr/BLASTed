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
	BJacobiRelaxation(const scalar reltol, const scalar abstol, const scalar divtol, 
			const bool checktol, const int maxiter);

	/// Carry out block-Jacobi relaxation
	void apply(const scalar *const b, scalar *const __restrict x) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;

	const scalar rtol;        ///< Relative Cauchy tolerance
	const scalar atol;        ///< Absolute Cauchy tolerance
	const scalar dtol;        ///< Cauchy tolerance for divergence (on the relative tolerance)
	const bool ctol;          ///< Whether to check for the Cauchy tolerances
	const int maxits;         ///< Maximum iterations - the only thing always obeyed
};

}

#endif
