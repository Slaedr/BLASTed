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
/// Column-oriented backward GS
const std::string cscbgsstr = "cscbgs";
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
	bool compute_factorization_res;       ///< Set to true if ILU residual computation is needed
};

template <typename scalar, typename index>
class FactoryBase
{
public:
	FactoryBase();
	virtual ~FactoryBase();

	/// Creates a preconditioner or relaxation object
	virtual SRPreconditioner<scalar,index>*
	create_preconditioner(const index ndim, const SolverSettings& settings) const = 0;

	/// Convert a string into a preconditioner type if possible.
	/// Should throw a invalid_argument if not possible.
	virtual BlastedSolverType solverTypeFromString(const std::string precstr) const = 0;
};

template <typename scalar, typename index>
class SRFactory : public FactoryBase<scalar,index>
{
public:
	/// Creates a preconditioner or relaxation object, for matrices stored by sparse rows
	/** \param ndim Number of rows in the matrix; the dimension of the problem
	 * \param settings Solver settings
	 * 
	 * Throws an instance of invalid_argument in case of invalid argument(s). */
	SRPreconditioner<scalar,index> *create_preconditioner(const index ndim,
	                                                      const SolverSettings& settings) const;

	/// Convert a string into a preconditioner type if possible. Throws a invalid_argument if not possible.
	BlastedSolverType solverTypeFromString(const std::string precstr) const;

private:
	template <int bs, StorageOptions stor>
	SRPreconditioner<scalar,index> *
	create_srpreconditioner_of_type(const int ndim, const AsyncSolverSettings& opts) const;
};

}
#endif
