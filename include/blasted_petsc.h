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
typedef enum {JACOBI, GS, SGS, ILU0, SAPILU0} Prec_type;

/** It has two operators - one for preconditioning and the other for relaxation.
 * The relaxation operator is only used when the local KSP is richardson. For all other local KSPs
 * (usually preonly), the preconditioning operator is used.
 */
struct Blasted_node
{
	void* bprec;              ///< BLASTed preconditioning object
	void *brelax;             ///< BLASTed relaxation object
	
	int bs;                   ///< Block size of dense blocks
	char *prectypestr;        ///< String identifier of the preconditioner type to use
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
	
	struct Blasted_node *next;  ///< Link to next Blasted context
};

/// State necessary for local preconditioners \sa Blasted_node	
typedef struct Blasted_node Blasted_data;

/// A list of BLASTed state objects for use with multiple BLASTed preconditioner instances
/** Eg. for use as multigrid smoothers.
 */
typedef struct
{
	Blasted_data *ctxlist;   ///< Linked list of BLASTed contexts for different instances
	int size;                ///< Size of the array

	double factorcputime;  ///< CPU time taken by factorizations by all BLASTed instances in this vector
	double factorwalltime; ///< Walltime taken by factorizations by all BLASTed instances in this vector
	double applycputime;   ///< CPU time taken by applications of all BLASTed instances in this vector
	double applywalltime;  ///< Walltime taken by applications of all BLASTed instances in this vector

} Blasted_data_list;

typedef struct Blasted_node Blasted_data;

/// Create a new list of BLASTed data contexts
Blasted_data_list newBlastedDataList();

/// Computes total time take by all BLASTed preconditioner instances in a context vector
void computeTotalTimes(Blasted_data_list *const bctv);

/// Destroy the list of BLASTed data contexts \warning Call only AFTER KSPDestroy.
/** Throws an instance of std::logic_error if the deletion fails.
 */
void destroyBlastedDataList(Blasted_data_list *const bdv);

/// Recursive function to set the BLASTed preconditioner wherever possible in the entire solver stack
/** Finds shell PCs and sets BLASTed as the preconditioner for each of them.
 * Assumptions:
 *  - If multigrid occurs, the same smoother (and the same number of sweeps) is used for both
 *   pre- and post-smoothing.
 *
 * \param ksp A PETSc solver context
 * \param bctx The BLASTed structures that store required settings and data; must be created by
 * \ref newBlastedDataList. It should later be deleted by the user, calling \ref destroyBlastedDataList
 *   after the ksp has been destroyed.
 */
PetscErrorCode setup_blasted_stack(KSP ksp, Blasted_data_list *const bctx);

/// Create a new BLASTed data context
Blasted_data newBlastedDataContext();

/// Adds a new node to the list of Blasted contexts
/** Adds the new node to the head of the list; ie, Blasted_data_list::ctxlist points to the new node
 * at the end of this function.
 * \param bdl The list to add a node to
 * \param bd The new node is initialized to a copy of this
 */
void appendBlastedDataContext(Blasted_data_list *const bdl, const Blasted_data bd);

/// Configure local PCs to enable BLASTed preconditioners
/** Instead of using this directly, consider using \ref setup_blasted_stack instead.
 * Note that it's not mandatory to use BLASTed preconditioners after this function is called;
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

/// Applies the local preconditioner by asynchronous iterations in parallel
/** \param pc is the PETSc local preconditioner context
 * \param r is the residual vector, ie, the RHS
 * \param z is the unknown vector to be computed.
 *
 * NOTE: It is assumed that the length of r and z on the local process is the same.
 */
PetscErrorCode apply_local_blasted(PC pc, Vec r, Vec z);

/// Applies a local asynchronous relaxation in parallel
PetscErrorCode relax_local_blasted(PC pc, Vec rhs, Vec x, Vec w, 
		PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt it, PetscBool guesszero, 
		PetscInt *outits, PCRichardsonConvergedReason *reason);

#ifdef __cplusplus
}
#endif

#endif
