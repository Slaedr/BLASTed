/** \file iterative_preconditioner.hpp
 * \brief Header for (local) iterative (possible asychronous) preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_ITERATIVE_PRECONDITIONER_H
#define BLASTED_ITERATIVE_PRECONDITIONER_H

#include "async_initialization_decl.hpp"
#include "solverops_base.hpp"
#include "solvertypes.h"

namespace blasted {

/// Parameters controlling iterative preconditioner generation (build/factorization) and application
struct IterPrecParams {
	bool usescaling;             ///< Whether to scale the matrix before building the preconditioner
	int thread_chunk_size;       ///< Number of threads in each OpenMP chunk
	bool threadedfactor;         ///< True for thread-parallel preconditioner build
	bool threadedapply;          ///< True for thread-parallel preconditioner application
	int nbuildsweeps;            ///< Number of iterations to build the preconditioner
	int napplysweeps;            ///< Number of iterations to apply the preconditioner
	FactInit factinittype;       ///< Type of initial value for preconditioner build
	ApplyInit applyinittype;     ///< Type of initial value for solution to preconditioner application
	BlastedIterType buildtype;   ///< Type of iteration to build preconditioner
	BlastedIterType applytype;   ///< Type of iteration to apply the preconditioner
};

/// Parameters controlling iterative preconditioner generation
struct BuildIterParams {
	bool usescaling;
	int thread_chunk_size;
	bool threaded;
	int nsweeps;
	FactInit inittype;
	BlastedIterType itertype;
};

/// Parameters controlling iterative preconditioner application
struct ApplyIterParams {
	bool usescaling;
	int thread_chunk_size;
	bool threaded;
	int nsweeps;
	ApplyInit inittype;
	BlastedIterType itertype;
};

template <typename scalar, typename index = int>
class IterativePreconditioner : public Preconditioner<scalar,index>
{
public:
	IterativePreconditioner(const StorageType storagetype)
		: Preconditioner<scalar,index>(storagetype)
	{ }

	void setAsyncParams(const IterPrecParams params) { iterparams = params; }

	IterPrecParams getParams() const { return iterparams; }

protected:
	IterPrecParams iterparams;
};

BuildIterParams extractBuildIterParams(const IterPrecParams params);

ApplyIterParams extractApplyIterParams(const IterPrecParams params);

}

#endif
