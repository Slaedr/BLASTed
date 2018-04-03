/** \file blasted_petsc.h
 * \brief C header for the PETSc interface of local preconditioning operations
 *
 * We only deal with local preconditioning operations, that is,
 * either single-process solves or the subdomain solves for a global solver such as
 * additive Schwarz.
 *
 * \author Aditya Kashi
 */

#ifndef BLASTED_PETSC_H
#define BLASTED_PETSC_H

#include <stdbool.h>
#include <petscksp.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The types of preconditioners that BLASTed provides
typedef enum {JACOBI, SGS, ILU0, SAPILU0} Prec_type;

/// State necessary for local preconditioners
/** The user must create a variable of this type,
 * set \ref first_setup_done to false and \ref bs to the required block size,
 * and then pass it to PCShellSetContext.
 */
typedef struct
{
	void* bmat;               ///< BLASTed matrix
	
	int bs;                   ///< Block size of dense blocks
	Prec_type prectype;       ///< The preconditioner to use
	int nbuildsweeps;         ///< Number of async build sweeps
	int napplysweeps;         ///< Number of async apply sweeps

	/// True if the initial one-time setup has been done
	/** MUST be set to false initially.
	 */
	bool first_setup_done;

	double cputime;           ///< Total CPU time taken by FGPILU
	double walltime;          ///< Total wall-clock time taken by FGPILU
	double factorcputime;     ///< CPU time taken for factorization
	double factorwalltime;    ///< Wall-clock time for factorization
	double applycputime;      ///< CPU time taken for application of the preconditioner
	double applywalltime;     ///< Wall-clock time for application
} Blasted_data;

/// Create a new BLASTed data context
Blasted_data newBlastedDataContext();

/// Configure local PCs to enable BLASTed preconditioners
/** Note that it's not mandatory to use BLASTed preconditioners after this function is called;
 * BLASTed preconditioners are only used in case the 'shell' preconditioner PCSHELL is requested.
 *
 * Adds two new command line options:
 * -> -blasted_pc_type [string: "jacobi", "sgs", "ilu0"]
 * -> -blasted_async_sweeps [array int[2]: number of build sweeps, number of apply sweeps]
 *
 * \param ksp The top level KSP or the global KSP at a multigrid level. Make sure:
 * - The KSP is created and set up.
 * - KSPSetOperators has been called to set the preconditioning matrix.
 * \param bctx The BLASTed structure that stores required settings and data which
 *   must be allocated before passing to this function. It should later be deleted by the user
 *   after the ksp has been destroyed.
 *
 * \warning bctx must NOT be deleted before the ksp is destroyed. Doing so will cause a memory leak.
 */
PetscErrorCode setup_localpreconditioner_blasted(KSP ksp, Blasted_data *const bctx);

/// Free arrays in the context struct
/** \param pc A PETSc subdomain preconditioner context
 */
PetscErrorCode cleanup_blasted(PC pc);

/// Update the local preconditioner for a new matrix
/** \param pc A PETSc subdomain preconditioner context
 */
PetscErrorCode compute_preconditioner_blasted(PC pc);

/// Applies the local preconditioner by Jacobi iterations in parallel
/** \param pc is the PETSc local preconditioner context
 * \param r is the residual vector, ie, the RHS
 * \param z is the unknown vector to be computed.
 *
 * NOTE: It is assumed that the length of r and z on the local process is the same.
 */
PetscErrorCode apply_local_blasted(PC pc, Vec r, Vec z);

/// Get timing data
/** \param pc A PETSc subdomain preconditioner context
 */
PetscErrorCode get_blasted_timing_data(PC pc, double *const factorcputime, 
		double *const factorwalltime, double *const applycputime, double *const applywalltime);

#ifdef __cplusplus
}
#endif

#endif
