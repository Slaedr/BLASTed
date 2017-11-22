/** \file blasted_petsc.h
 * \brief C header for the PETSc interface of preconditioning operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_PETSC_H
#define BLASTED_PETSC_H

#include <stdbool.h>
#include <petscpc.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The types of preconditioners that BLASTed provides
typedef enum {JACOBI, SGS, ILU0} Prec_type;

/// State necessary for preconditioners
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

/// Free arrays in the context struct
PetscErrorCode cleanup_blasted(PC pc);

/// Update the preconditioner for a new matrix, if required
PetscErrorCode compute_preconditioner_blasted(PC pc);

/// Applies the preconditioner by Jacobi iterations in parallel
/** \param pc is the PETSc preconditioner context
 * \param r is the residual vector, ie, the RHS
 * \param z is the unknown vector to be computed.
 *
 * NOTE: It is assumed that the length of r and z on the local process is the same.
 */
PetscErrorCode apply_local_blasted(PC pc, Vec r, Vec z);

/// Get timing data
PetscErrorCode get_blasted_timing_data(PC pc, double *const factorcputime, 
		double *const factorwalltime, double *const applycputime, double *const applywalltime);

#ifdef __cplusplus
}
#endif

#endif
