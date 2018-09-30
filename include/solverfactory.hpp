/** \file solverfactory.hpp
 * \brief Functions to return a preconditioner or relaxation object from runtime parameters
 * \author Aditya Kashi
 * \date 2018-04
 */

#ifndef BLASTED_SOLVERFACTORY_H
#define BLASTED_SOLVERFACTORY_H

#include <string>

#include "solvertypes.h"
#include "async_initialization_decl.hpp"
#include "solverops_base.hpp"

namespace blasted {

/** \defgroup prectypelabels Strings for selecting the basic preconditioning method
 * @{
 */
/// Identity
const std::string noprecstr = "none";
/// Jacobi
const std::string jacobistr = "jacobi";
/// Gauss-Seidel
const std::string gsstr = "gs";
/// SGS
const std::string sgsstr = "sgs";
/// ILU(0)
const std::string ilu0str = "ilu0";
/// Sequentially applied ILU(0)
const std::string sapilu0str = "sapilu0";
/** @} */

/// Basic settings needed for most iterations
struct SolverSettings {
	BlastedSolverType prectype;           ///< The type of preconditioner to use
	int bs;                               ///< Size of small dense blocks in the system matrix
	StorageOptions blockstorage;          ///< Layout within individual blocks - RowMajor or ColMajor
	/// Set to true if relaxation is desired instead of preconditioning
	/** This is not possible for some methods, in which case preconditioning will be used anyway.
	 */
	bool relax;
	int thread_chunk_size;                ///< Number of work-items (iterations) in each thread chunk
};

/// Settings needed for most asynchronous iterations
struct AsyncSolverSettings : public SolverSettings {
	int nbuildsweeps;                     ///< Number of build sweeps
	int napplysweeps;                     ///< Number of apply sweeps
	FactInit fact_inittype;               ///< Initialization type for asynchronous factorization
	ApplyInit apply_inittype;             ///< Initialization type for asynchronous triangular solves
};

/// Creates a preconditioner or relaxation object, for matrices stored by sparse rows
/** \param ndim Number of rows in the matrix; the dimension of the problem
 * \param settings Solver settings
 * 
 * Throws an instance of invalid_argument in case of invalid argument(s). */
template <typename scalar, typename index>
SRPreconditioner<scalar,index> *create_sr_preconditioner(const index ndim,
                                                         const SolverSettings& settings);
	// (const std::string precstr, const int bs, const std::string blockstorage,
	//  const bool relaxation,
	//  const std::map<std::string,int>& intParamList,
	//  const std::map<std::string,double>& floatParamList);

/// Convert a string into a preconditioner type if possible. Throws a invalid_argument if not possible.
BlastedSolverType solverTypeFromString(const std::string precstr);

}
#endif
