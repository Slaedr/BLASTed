/** \file
 * \brief Testing functions that use PETSc and don't use C++
 */

#ifndef BLASTED_TESTING_PETSC_H
#define BLASTED_TESTING_PETSC_H

#include <petscmat.h>

#ifdef __cplusplus
extern "C" {
#endif

/// An object representing a discrete linear problem
typedef struct {
	Mat lhs;                     ///< LHS operator
	Vec b;                       ///< RHS vector (load vector)
	Vec uexact;                  ///< Exact solution vector, if available
} DiscreteLinearProblem ;

/// Destroys the linear problem
int destroyDiscreteLinearProblem(DiscreteLinearProblem *const dlp);

/// Reads a matrix, RHS vector and exact solution from PETSc binary files
int readLinearSystemFromFiles(const char *const matfile, const char *const bfile, const char *const xfile,
                              DiscreteLinearProblem *const lp);

/// Computes the vector 2-norm of the difference u1 - u2, scaled by the square-root of their size
int compute_difference_norm(const Vec u1, const Vec u2, PetscReal *const errnorm);

/// Compares a solver with a reference solver (usually from PETSc) in terms of some criteria
/** PETSc options controlling the criteria are
 * -test_type <string> -error_tolerance <float>
 * where -test_type can be compare_its, issame or upper_bound_its.
 * Note that test type 'issame' compares the solution vectors obtained from the two solvers,
 * along with the number of iterations they take (both use the same relative tolerance currently).
 * If anything but these is specified, such as 'convergence', we only test whether the solver converged.
 */
int compareSolverWithRef(const int refkspiters, const int avgkspiters, Vec uref, Vec u);

/// Runs a PETSc solver and a BLASTed solver from command line settings
int runComparisonVsPetsc(const DiscreteLinearProblem lp);

/// Run a BLASTed solver in a C++ function so as to test some C++-only interface features
/** This function can be called from C.
 * \warning This is not exactly equivalent to the C version \ref runComparisonVsPetsc.
 */
int runComparisonVsPetsc_cpp(const DiscreteLinearProblem lp);

/// Returns the block size of the matrix if it uses a block format, else returns 1
int getBlockSize(const Mat A);

#ifdef __cplusplus
}
#endif

#endif
