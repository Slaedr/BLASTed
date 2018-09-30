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
#include "relaxation_jacobi.hpp"
#include "relaxation_chaotic.hpp"
#include "relaxation_async_sgs.hpp"

namespace blasted {

BlastedSolverType solverTypeFromString(const std::string precstr2)
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
template <typename scalar, typename index, int bs, StorageOptions stor>
static
SRPreconditioner<scalar,index> *create_srpreconditioner_of_type(const int ndim,
                                                                const AsyncSolverSettings& opts)
{
	if(opts.prectype == BLASTED_JACOBI)
		if(opts.relax)
			return new BJacobiRelaxation<scalar,index,bs,stor>();
		else
			return new BJacobiSRPreconditioner<scalar,index,bs,stor>();
	else if(opts.prectype == BLASTED_GS) {
		if(!opts.relax) {
			std::cout << "WARNING: SolverFactory: GS preconditioner not yet implemented.";
			std::cout << " Using the relaxation instead.\n";
		}
		return new ChaoticBlockRelaxation<scalar,index,bs,stor>(opts.thread_chunk_size);
	}
	else if(opts.prectype == BLASTED_SGS) {
		if(opts.relax) {
			return new AsyncBlockSGS_Relaxation<scalar,index,bs,stor>(opts.thread_chunk_size);
		}
		else
			return new AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>
				(opts.napplysweeps, opts.apply_inittype, opts.thread_chunk_size);
	}
	else if(opts.prectype == BLASTED_ILU0) {
		if(opts.relax) {
			std::cout << "Solverfactory: ILU relaxation is not implemented.";
			std::cout << " Using the preconditioner.\n";
		}
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(opts.nbuildsweeps, opts.napplysweeps, opts.thread_chunk_size,
			 opts.fact_inittype, opts.apply_inittype, true, true);
	}
	else if(opts.prectype == BLASTED_SAPILU0) {
		if(opts.relax) {
			std::cout << "Solverfactory: ILU relaxation is not implemented.";
			std::cout << " Using the preconditioner.\n";
		}
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(opts.nbuildsweeps, opts.napplysweeps, opts.thread_chunk_size,
			 opts.fact_inittype, opts.apply_inittype, true, false);
	}
	else if(opts.prectype == BLASTED_NO_PREC) {
		if(opts.relax) {
			std::cout << "WARNING: Solverfactory: ILU relaxation is not possible.";
			std::cout << " Using the preconditioner instead.\n";
		}
		return new NoPreconditioner<scalar,index>(ndim);
	}
	else
		throw std::invalid_argument("Invalid preconditioner!");
}

template <typename scalar, typename index>
SRPreconditioner<scalar,index> *create_sr_preconditioner(const index ndim, const SolverSettings& set)
{
	SRPreconditioner<scalar,index> *p = nullptr;

	const AsyncSolverSettings& opts = reinterpret_cast<const AsyncSolverSettings&>(set);
		
	if(opts.bs == 1) {
		if(opts.prectype == BLASTED_JACOBI) {
			if(opts.relax)
				p = new JacobiRelaxation<scalar,index>();
			else
				p = new JacobiSRPreconditioner<scalar,index>();
		}
		else if(opts.prectype == BLASTED_GS) {
			p = new ChaoticRelaxation<scalar,index>(opts.thread_chunk_size);
			if(!opts.relax) {
				std::cout << "solverfactory(): Warning: Forward Gauss-Seidel preconditioner ";
				std::cout << "is not implemented; using relaxation instead.\n";
			}
		}
		else if(opts.prectype == BLASTED_SGS) {
			if(opts.relax)
				p = new AsyncSGS_Relaxation<scalar,index>(opts.thread_chunk_size);
			else
				p = new AsyncSGS_SRPreconditioner<scalar,index>
					(opts.napplysweeps, opts.apply_inittype, opts.thread_chunk_size);
		}
		else if(opts.prectype == BLASTED_ILU0)
			p = new AsyncILU0_SRPreconditioner<scalar,index>
				(opts.nbuildsweeps, opts.napplysweeps,
				 opts.thread_chunk_size,
				 opts.fact_inittype, opts.apply_inittype, true,true);
		else if(opts.prectype == BLASTED_SAPILU0)
			return new AsyncILU0_SRPreconditioner<scalar,index>
				(opts.nbuildsweeps, opts.napplysweeps,
				 opts.thread_chunk_size,
				 opts.fact_inittype, opts.apply_inittype, true,false);
		else if(opts.prectype == BLASTED_NO_PREC)
			return new NoPreconditioner<scalar,index>(ndim);
		else
			throw std::invalid_argument("Invalid preconditioner!");
	}
	else if(opts.blockstorage == RowMajor) 
	{
		if(opts.bs == 4) {
			p = create_srpreconditioner_of_type<scalar,index,4,RowMajor>(ndim,opts);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(opts.bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<scalar,index,BUILD_BLOCK_SIZE,RowMajor>(ndim,opts);
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
			p = create_srpreconditioner_of_type<scalar,index,4,ColMajor>(ndim,opts);
		else if(opts.bs == 5) {
			p = create_srpreconditioner_of_type<scalar,index,5,ColMajor>(ndim,opts);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(opts.bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<scalar,index,BUILD_BLOCK_SIZE,ColMajor>(ndim,opts);
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

template SRPreconditioner<double,int>* create_sr_preconditioner<double,int>
(const int ndim, const SolverSettings& opts);

}
