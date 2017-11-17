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
typedef struct
{
	const void* bmat;               ///< BLASTed matrix
	
	const int bs;                   ///< Block size of dense blocks
	const Prec_type prectype;       ///< The preconditioner to use
	const int nbuildsweeps;         ///< Number of async build sweeps
	const int napplysweeps;         ///< Number of async apply sweeps
	
	double cputime;           ///< Total CPU time taken by FGPILU
	double walltime;          ///< Total wall-clock time taken by FGPILU
	double factorcputime;     ///< CPU time taken for factorization
	double factorwalltime;    ///< Wall-clock time for factorization
	double applycputime;      ///< CPU time taken for application of the preconditioner
	double applywalltime;     ///< Wall-clock time for application
} Blasted_data;

/// Set up the preconditioner
/** Queries the PETSc options database for the information required.
 *
 * \param[in,out] pc The PETSc preconditioner context
 * \param[in] blocksize The size of small dense blocks that make up the matrix
 */
PetscErrorCode setup_blasted(PC pc, const int blocksize);

/// Free arrays in the context struct
PetscErrorCode cleanup_blasted(PC pc);

/// Update the preconditioner for a new matrix, if required
PetscErrorCode compute_preconditioner(PC pc);

/// Applies the preconditioner by Jacobi iterations in parallel
/** \param pc is the PETSc preconditioner context
 * \param r is the residual vector, ie, the RHS
 * \param z is the unknown vector to be computed.
 *
 * NOTE: It is assumed that the length of r and z on the local process is the same.
 */
PetscErrorCode apply_local(PC pc, Vec r, Vec z);

#ifdef __cplusplus
}
#endif

#endif
