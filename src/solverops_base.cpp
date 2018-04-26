/** \file preconditioners.cpp
 * \brief Implementation of some common functionality for preconditioners
 * \author Aditya Kashi
 */

#include "preconditioners.hpp"

namespace blasted {

template <typename scalar, typename index>
Preconditioner<scalar,index>::Preconditioner(const StorageType stype)
	: _type{stype}
{ }

template <typename scalar, typename index>
Preconditioner<scalar,index>::~Preconditioner()
{ }

template <typename scalar, typename index>
SRPreconditioner<scalar,index>::SRPreconditioner()
	: Preconditioner<scalar,index>(SPARSEROW), mat{nullptr, nullptr, nullptr, nullptr, 0}

template <typename scalar, typename index>
void SRPreconditioner<scalar,index>::wrap(const index n_brows, const index *const brptrs,
	const index *const bcinds, const scalar *const values, const index *const dinds)
{
	mat.nbrows = n_brows;
	mat.browptr = brptrs;
	mat.bcolind = bcinds;
	mat.vals = values;
	mat.diagind = dinds;

	// Since the matrix has been changed, recompute the preconditioner.
	// This should call the compute function from the appropriate derived class
	compute();
}

// instantiations
template SRPreconditioner<double,int>;

}
