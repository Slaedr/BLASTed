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
SRPreconditioner<scalar,index>::SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix)
	: Preconditioner<scalar,index>(SPARSEROW), pmat(std::move(matrix)),
	  mat(&pmat.browptr[0], &pmat.bcolind[0], &pmat.vals[0], &pmat.diagind[0], &pmat.browendptr[0],
	      pmat.nbrows, pmat.nnzb, pmat.nbstored)
{ }

template <typename scalar, typename index>
NoPreconditioner<scalar,index>::NoPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
                                                 const index bs)
	: SRPreconditioner<scalar,index>(std::move(matrix)), ndim{pmat.nbrows*bs}
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
