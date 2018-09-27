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

/// Creates the correct preconditioner or relaxation from the arguments for the template
/** Note that if a relaxation is requested for an algorithm got which relaxation is not implemented,
 * the corresponding preconditioner is returned instead after printing a warning.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
static inline
SRPreconditioner<scalar,index> *create_srpreconditioner_of_type
	(const std::string precstr, const bool relax,
	 const std::map<std::string,int>& intlist, const std::map<std::string,double>& floatlist)
{
	if(precstr == jacobistr)
		if(relax)
			return new BJacobiRelaxation<scalar,index,bs,stor>();
		else
			return new BJacobiSRPreconditioner<scalar,index,bs,stor>();
	else if(precstr == gsstr) {
		if(!relax) {
			std::cout << "WARNING: SolverFactory: GS preconditioner not yet implemented.";
			std::cout << " Using the relaxation instead.\n";
		}
		return new ChaoticBlockRelaxation<scalar,index,bs,stor>();
	}
	else if(precstr == sgsstr) {
		if(relax) {
			return new AsyncBlockSGS_Relaxation<scalar,index,bs,stor>();
		}
		else
			return new AsyncBlockSGS_SRPreconditioner<scalar,index,bs,stor>(intlist.at(napplysweeps));
	}
	else if(precstr == ilu0str) {
		if(relax) {
			std::cout << "Solverfactory: ILU relaxation is not implemented.";
			std::cout << " Using the preconditioner.\n";
		}
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(intlist.at(nbuildsweeps), intlist.at(napplysweeps), intlist.at(thread_chunk_size),
			 intlist.at(fact_inittype), intlist.at(apply_inittype), true, true);
	}
	else if(precstr == sapilu0str) {
		if(relax) {
			std::cout << "Solverfactory: ILU relaxation is not implemented.";
			std::cout << " Using the preconditioner.\n";
		}
		return new AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>
			(intlist.at(nbuildsweeps), intlist.at(napplysweeps), intlist.at(thread_chunk_size),
			 intlist.at(fact_inittype), intlist.at(apply_inittype), true, false);
	}
	else if(precstr == noprecstr) {
		if(relax) {
			std::cout << "WARNING: Solverfactory: ILU relaxation is not possible.";
			std::cout << " Using the preconditioner instead.\n";
		}
		return new NoPreconditioner<scalar,index>(intlist.at(ndimstr));
	}
	else
		throw std::invalid_argument("Invalid preconditioner!");
}

template <typename scalar, typename index>
SRPreconditioner<scalar,index> *create_sr_preconditioner
    (const std::string precstr, const int bs,
     const std::string blockstorage, const bool relax,
     const std::map<std::string,int>& intParamList,
     const std::map<std::string,double>& floatParamList)
{
	SRPreconditioner<scalar,index> *p = nullptr;
		
	if(bs == 1) {
		if(precstr == jacobistr) {
			if(relax)
				p = new JacobiRelaxation<scalar,index>();
			else
				p = new JacobiSRPreconditioner<scalar,index>();
		}
		else if(precstr == gsstr) {
			p = new ChaoticRelaxation<scalar,index>();
			if(!relax) {
				std::cout << "solverfactory(): Warning: Forward Gauss-Seidel preconditioner ";
				std::cout << "is not implemented; using relaxation instead.\n";
			}
		}
		else if(precstr == sgsstr) {
			if(relax)
				p = new AsyncSGS_Relaxation<scalar,index>();
			else
				p = new AsyncSGS_SRPreconditioner<scalar,index>(intParamList.at(napplysweeps));
		}
		else if(precstr == ilu0str)
			p = new AsyncILU0_SRPreconditioner<scalar,index>
				(intParamList.at(nbuildsweeps), intParamList.at(napplysweeps),
				 intParamList.at(thread_chunk_size),
				 intParamList.at(fact_inittype), intParamList.at(apply_inittype), true,true);
		else if(precstr == sapilu0str)
			return new AsyncILU0_SRPreconditioner<scalar,index>
				(intParamList.at(nbuildsweeps), intParamList.at(napplysweeps),
				 intParamList.at(thread_chunk_size),
				 intParamList.at(fact_inittype), intParamList.at(apply_inittype), true,false);
		else if(precstr == noprecstr)
			return new NoPreconditioner<scalar,index>(intParamList.at(ndimstr));
		else
			throw std::invalid_argument("Invalid preconditioner!");
	}
	else if(blockstorage == rowmajorstr) 
	{
		if(bs == 4) {
			p = create_srpreconditioner_of_type<scalar,index,4,RowMajor>(precstr, relax,
					intParamList, floatParamList);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<scalar,index,BUILD_BLOCK_SIZE,RowMajor>(precstr, 
				      relax, intParamList, floatParamList);
		}
#endif
		else {
			throw std::invalid_argument("Block size " + std::to_string(bs) + 
					" not supported for row major!");
		}
	}
	else if(blockstorage == colmajorstr)
	{
		if(bs==4)
			p = create_srpreconditioner_of_type<scalar,index,4,ColMajor>(precstr, relax,
					intParamList, floatParamList);
		else if(bs == 5) {
			p = create_srpreconditioner_of_type<scalar,index,5,ColMajor>(precstr, relax,
					intParamList, floatParamList);
		}
#ifdef BUILD_BLOCK_SIZE
		else if(bs == BUILD_BLOCK_SIZE) {
			p = create_srpreconditioner_of_type<scalar,index,BUILD_BLOCK_SIZE,ColMajor>(precstr, 
				      relax, intParamList, floatParamList);
		}
#endif
		else {
			throw std::invalid_argument("Block size " + std::to_string(bs) + 
					" not supported for column major!");
		}
	}
	else {
		throw std::invalid_argument("Block ordering must be either " + rowmajorstr + " or " +
				colmajorstr);
	}

	return p;
}

template SRPreconditioner<double,int>* create_sr_preconditioner<double,int>
( const std::string precstr, const int bs, const std::string blockstorage, const bool relax,
  const std::map<std::string,int>& intParamList, const std::map<std::string,double>& floatParamList);

}
