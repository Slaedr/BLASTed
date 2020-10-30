/** \file iterative_preconditioner.hpp
 * \brief Header for (local) iterative (possible asychronous) preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_ITERATIVE_PRECONDITIONER_H
#define BLASTED_ITERATIVE_PRECONDITIONER_H

#include "async_initialization_decl.hpp"
#include "solverops_base.hpp"

namespace blasted {

struct AsyncParams {
	bool usescaling;             ///< Whether to scale the matrix before building the preconditioner
	bool threadedfactor;         ///< True for thread-parallel preconditioner build
	bool threadedapply;          ///< True for thread-parallel preconditioner application
	int nbuildsweeps;            ///< Number of iterations to build the preconditioner
	int napplysweeps;            ///< Number of iterations to apply the preconditioner
	int thread_chunk_size;       ///< Number of threads in each OpenMP chunk
	FactInit factinittype;       ///< Type of initial value for preconditioner build
	ApplyInit applyinittype;     ///< Type of initial value for solution to preconditioner application
};

template <typename scalar, typename index = int>
class IterativePreconditioner : public Preconditioner<scalar,index>
{
public:
	IterativePreconditioner(const StorageType storagetype)
		: Preconditioner<scalar,index>(storagetype)
	{ }

	void setAsyncParams(const AsyncParams params) { asyncparams = params; }

	AsyncParams getAsyncParams() const { return asyncparams; }

protected:
	AsyncParams asyncparams;
};

}

#endif
