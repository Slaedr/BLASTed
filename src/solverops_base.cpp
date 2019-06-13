/** \file solverops_base.cpp
 * \brief Implementation of some common functionality for preconditioners
 * \author Aditya Kashi
 */

#include "solverops_base.hpp"

namespace blasted {

template <typename scalar, typename index>
Preconditioner<scalar,index>::Preconditioner(const StorageType stype)
	: AbstractLinearOperator<scalar,index>(stype)
{ }

template <typename scalar, typename index>
Preconditioner<scalar,index>::~Preconditioner()
{ }

template <typename scalar, typename index>
SRPreconditioner<scalar,index>::SRPreconditioner()
	: Preconditioner<scalar,index>(SPARSEROW)
{ }

template <typename scalar, typename index>
void SRPreconditioner<scalar,index>::wrap(const index n_brows, const index *const brptrs,
                                          const index *const bcinds, const scalar *const values,
                                          const index *const dinds)
{
	mat.nbrows = n_brows;
	mat.browptr = brptrs;
	mat.bcolind = bcinds;
	mat.vals = values;
	mat.diagind = dinds;
	if(n_brows > 0)
		mat.browendptr = &brptrs[1];
}

template <typename scalar, typename index>
NoPreconditioner<scalar,index>::NoPreconditioner(const index matrixdim)
	: SRPreconditioner<scalar,index>(), ndim{matrixdim}
{ }

template <typename scalar, typename index>
void NoPreconditioner<scalar,index>::apply(const scalar *const x, scalar *const __restrict y) const
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < ndim; i++)
		y[i] = x[i];
}

template <typename scalar, typename index>
void NoPreconditioner<scalar,index>::apply_relax(const scalar *const x, scalar *const __restrict y) const
{
}

// instantiations
template class Preconditioner<double,int>;
template class SRPreconditioner<double,int>;
template class NoPreconditioner<double,int>;

}
