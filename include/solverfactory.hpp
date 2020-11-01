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
/// Sequential ILU(0)
const std::string seqilu0str = "seqilu0";
/// Sequentially factored ILU(0)
const std::string sfilu0str = "sfilu0";
/// Sequentially applied ILU(0)
const std::string sapilu0str = "sapilu0";
/// ILU(0) by Jacobi iterations rather than asynchronous
const std::string jacilu0str = "jacilu0";
/// Column-oriented backward GS
const std::string cscbgsstr = "cscbgs";
/// Level-scheduled SGS
const std::string levelsgsstr = "level_sgs";
/// Asynchronous factorized ILU0 with level-scheduled application
const std::string asynclevelilustr = "async_level_ilu0";
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

	/// Default destructor
	virtual ~SolverSettings() = default;
};

/// Settings needed for most asynchronous iterations
struct AsyncSolverSettings : public SolverSettings {
	bool scale;                           ///< Use the symmetrically scaled matrix instead of the original
	int nbuildsweeps;                     ///< Number of build sweeps
	int napplysweeps;                     ///< Number of apply sweeps
	FactInit fact_inittype;               ///< Initialization type for asynchronous factorization
	ApplyInit apply_inittype;             ///< Initialization type for asynchronous triangular solves
	bool compute_precinfo;                ///< Set to true if extra information is needed
};

template <typename scalar, typename index>
class FactoryBase
{
public:
	FactoryBase();
	virtual ~FactoryBase();

	/// Creates a preconditioner or relaxation object
	virtual SRPreconditioner<scalar,index>*
	create_preconditioner(SRMatrixStorage<const scalar, const index>&& prec_matrix,
	                      const SolverSettings& settings) const = 0;

	/// Convert a string into a preconditioner type if possible.
	/// Should throw a invalid_argument if not possible.
	virtual BlastedSolverType solverTypeFromString(const std::string precstr) const = 0;
};

template <typename scalar, typename index>
class SRFactory : public FactoryBase<scalar,index>
{
public:
	/// Creates a preconditioner or relaxation object, for matrices stored by sparse rows
	/** \param prec_matrix The matrix from which to construct the preconditioner. The output
	 *   SRPreconditioner will store a view of this matrix. This object gets invalidated, so
	 *   do not use the object after calling this fucntion.
	 * \param settings Solver settings
	 *
	 * The output preconditioner context stores a reference to the given prec_matrix.
	 * Throws an instance of std::invalid_argument in case of invalid argument(s).
	 */
	SRPreconditioner<scalar,index> *
	create_preconditioner(SRMatrixStorage<const scalar, const index>&& prec_matrix,
	                      const SolverSettings& settings) const;

	/// Convert a string into a preconditioner type if possible. Throws a invalid_argument if not possible.
	BlastedSolverType solverTypeFromString(const std::string precstr) const;

private:
	template <int bs, StorageOptions stor>
	SRPreconditioner<scalar,index> *
	create_srpreconditioner_of_type(SRMatrixStorage<const scalar, const index>&& prec_matrix,
	                                const AsyncSolverSettings& opts) const;
};

}
#endif
