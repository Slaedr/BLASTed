/** \file solverfactory.cpp
 * \brief Functions to return a preconditioner or relaxation object from runtime parameters
 * \author Aditya Kashi
 * \date 2018-04
 */

#include <stdexcept>
#include <iostream>
#include "solverfactory.hpp"
#include "solverops_jacobi.hpp"
#include "solverops_sgs.hpp"
#include "solverops_ilu0.hpp"
#include "relaxation_chaotic.hpp"
#include "solverops_levels_sgs.hpp"
#include "solverops_levels_ilu0.hpp"

namespace blasted {

const std::array<std::string,6> PrecInfoList::descr = {"factor_remainder", "factor_init_rem",
                                                       "upperF_min_dgdom", "upperF_avg_dgdom",
                                                       "lowerF_min_dgdom", "lowerF_avg_dgdom"};
const int PrecInfoList::field_width = 18;

template <typename scalar, typename index>
FactoryBase<scalar,index>::FactoryBase()
{ }

template <typename scalar, typename index>
FactoryBase<scalar,index>::~FactoryBase()
{ }

template class FactoryBase<double,int>;

template <typename scalar, typename index>
BlastedSolverType SRFactory<scalar,index>::solverTypeFromString(const std::string precstr2) const
{
	BlastedSolverType ptype;
	if(precstr2 == jacobistr)
		ptype = BLASTED_JACOBI;
	else if(precstr2 == gsstr)
		ptype = BLASTED_GS;
	else if(precstr2 == sgsstr)
		ptype = BLASTED_SGS;
	else if(precstr2 == ilu0str)
		ptype = BLASTED_ILU0;
	else if(precstr2 == sapilu0str)
		ptype = BLASTED_SAPILU0;
	else if(precstr2 == cscbgsstr)
		ptype = BLASTED_CSC_BGS;
	else if(precstr2 == levelsgsstr)
		ptype = BLASTED_LEVEL_SGS;
	else if(precstr2 == asynclevelilustr)
		ptype = BLASTED_ASYNC_LEVEL_ILU0;
	else if(precstr2 == noprecstr)
		ptype = BLASTED_NO_PREC;
	else {
		throw std::invalid_argument("BLASTed: Preconditioner type not available!");
	}
	return ptype;
}

/// Creates the correct preconditioner or relaxation from the arguments for the template
/** Note that if a relaxation is requested for an algorithm got which relaxation is not implemented,
 * the corresponding preconditioner is returned instead after printing a warning.
 */
template <typename scalar, typename index>
template <int bs, StorageOptions stor>
SRPreconditioner<scalar,index>*
SRFactory<scalar,index>
::create_srpreconditioner_of_type(SRMatrixStorage<const scalar,const index>&& mat,
                                  const AsyncSolverSettings& opts) const
{
	if(opts.prectype == BLASTED_JACOBI) {
		return new BJacobiSRPreconditioner<scalar,index,bs,stor>(std::move(mat));
	}
	else if(opts.prectype == BLASTED_GS) {
		return new ChaoticBlockRelaxation<scalar,index,bs,stor>(std::move(mat), opts.napplysweeps,
		                                                        opts.thread_chunk_size);
	}
	else if(opts.prectype == BLASTED_SGS) {
		return new AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>
			(std::move(mat), opts.napplysweeps, opts.apply_inittype, opts.thread_chunk_size);
	}
	else if(opts.prectype == BLASTED_ILU0) {
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(std::move(mat), opts.nbuildsweeps, opts.napplysweeps, opts.scale, opts.thread_chunk_size,
			 opts.fact_inittype, opts.apply_inittype, true, true, opts.compute_precinfo);
	}
	else if(opts.prectype == BLASTED_SAPILU0) {
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(std::move(mat), opts.nbuildsweeps, opts.napplysweeps, opts.scale, opts.thread_chunk_size,
			 opts.fact_inittype, opts.apply_inittype, true, false, opts.compute_precinfo);
	}
	else if(opts.prectype == BLASTED_LEVEL_SGS) {
		return new Level_BSGS<scalar,index,bs,stor>(std::move(mat));
	}
	else if(opts.prectype == BLASTED_ASYNC_LEVEL_ILU0) {
		return new Async_Level_BlockILU0<scalar,index,bs,stor>(std::move(mat),opts.nbuildsweeps,
		                                                       opts.scale,
		                                                       opts.thread_chunk_size,
		                                                       opts.fact_inittype, true,
		                                                       opts.compute_precinfo);
	}
	else if(opts.prectype == BLASTED_NO_PREC) {
		return new NoPreconditioner<scalar,index>(std::move(mat), bs);
	}
	else
		throw std::invalid_argument("Invalid preconditioner!");
}

/** Right now, this factory fails if mat actually owns its storage because it gets destroyed at the
 * end of this function. But, eventually it'll be moved to the preconditioners' SRMatrixStorages and
 * it should be fine.
 */
template <typename scalar, typename index>
SRPreconditioner<scalar,index>*
SRFactory<scalar,index>::create_preconditioner(SRMatrixStorage<const scalar, const index>&& mat,
                                               const SolverSettings& set) const
{
	SRPreconditioner<scalar,index> *p = nullptr;

	const AsyncSolverSettings& opts = dynamic_cast<const AsyncSolverSettings&>(set);

	if(opts.bs == 1) {
		if(opts.prectype == BLASTED_JACOBI) {
			p = new JacobiSRPreconditioner<scalar,index>(std::move(mat));
		}
		else if(opts.prectype == BLASTED_GS) {
			p = new ChaoticRelaxation<scalar,index>(std::move(mat), opts.napplysweeps,
			                                        opts.thread_chunk_size);
		}
		else if(opts.prectype == BLASTED_CSC_BGS) {
			p = new CSC_BGS_Preconditioner<scalar,index>(std::move(mat), opts.napplysweeps,
			                                             opts.thread_chunk_size);
		}
		else if(opts.prectype == BLASTED_SGS) {
			p = new AsyncSGS_SRPreconditioner<scalar,index>
				(std::move(mat), opts.napplysweeps, opts.apply_inittype, opts.thread_chunk_size);
		}
		else if(opts.prectype == BLASTED_LEVEL_SGS) {
			p = new Level_SGS<scalar,index>(std::move(mat));
		}
		else if(opts.prectype == BLASTED_ILU0) {
			p = new AsyncILU0_SRPreconditioner<scalar,index>
				(std::move(mat), opts.nbuildsweeps, opts.napplysweeps,
				 opts.scale, opts.thread_chunk_size,
				 opts.fact_inittype, opts.apply_inittype, opts.compute_precinfo, true,true);
		}
		else if(opts.prectype == BLASTED_SAPILU0) {
			p = new AsyncILU0_SRPreconditioner<scalar,index>
				(std::move(mat), opts.nbuildsweeps, opts.napplysweeps,
				 opts.scale, opts.thread_chunk_size,
				 opts.fact_inittype, opts.apply_inittype, opts.compute_precinfo, true,false);
		}
		else if(opts.prectype == BLASTED_ASYNC_LEVEL_ILU0) {
			p = new Async_Level_ILU0<scalar,index>(std::move(mat), opts.nbuildsweeps, opts.scale,
			                                       opts.thread_chunk_size, opts.fact_inittype, true,
			                                       opts.compute_precinfo);
		}
		else if(opts.prectype == BLASTED_NO_PREC) {
			p = new NoPreconditioner<scalar,index>(std::move(mat), 1);
		}
		else
			throw std::invalid_argument("Invalid preconditioner!");
	}
	else if(opts.blockstorage == RowMajor)
	{
		if(opts.bs == 4) {
			p = create_srpreconditioner_of_type<4,RowMajor>(std::move(mat),opts);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(opts.bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<scalar,index,BUILD_BLOCK_SIZE,RowMajor>(std::move(mat),
			                                                                            opts);
		}
#endif
		else {
			throw std::invalid_argument("Block size " + std::to_string(opts.bs) +
					" not supported for row major!");
		}
	}
	else if(opts.blockstorage == ColMajor)
	{
		if(opts.bs==4)
			p = create_srpreconditioner_of_type<4,ColMajor>(std::move(mat),opts);
		else if(opts.bs == 5) {
			p = create_srpreconditioner_of_type<5,ColMajor>(std::move(mat),opts);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(opts.bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<BUILD_BLOCK_SIZE,ColMajor>(std::move(mat),opts);
		}
#endif
		else {
			throw std::invalid_argument("Block size " + std::to_string(opts.bs) +
					" not supported for column major!");
		}
	}
	else {
		throw std::invalid_argument("Block ordering must be either rowmajor or colmajor!");
	}

	return p;
}

template class SRFactory<double,int>;

}
