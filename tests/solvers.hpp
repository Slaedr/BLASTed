/** \file solvers.hpp
 * \brief Definitions of some solver routines
 * \author Aditya Kashi
 * \date 2017-11
 */

#ifndef BLASTED_SOLVERS_H
#define BLASTED_SOLVERS_H

#include "solverops_base.hpp"
#include "blockmatrices.hpp"

namespace blasted {

typedef int a_int;
typedef double a_real;

/// Information about a solve that was performed
struct SolveInfo {
	bool converged{};                 ///< Whether the solve converged to required tolerance
	int iters{};                      ///< Number of iterations performed
	a_real resnorm{};                 ///< 2-norm of the final residual vector
	a_real bnorm{};                   ///< 2-Norm of RHS
	double walltime{};                ///< Total wall-time taken by the solver
	double cputime{};                 ///< Total CPU time taken by the solver
	double precapplywtime{};          ///< Total wall time taken by preconditioner application
};

/// Abstract preconditioned iterative solver
class IterativeSolverBase
{
protected:
	int maxiter;                                  ///< Max number of iterations
	double tol;                                   ///< Tolerance
	mutable double walltime;                      ///< Stores wall-clock time measurement of solver
	mutable double cputime;                       ///< Stores CPU time measurement of the solver

public:
	IterativeSolverBase();

	virtual ~IterativeSolverBase();
	
	/// Set tolerance and max iterations
	void setParams(const double toler, const int maxits);
	
	/// Sets time accumulators to zero
	void resetRunTimes();

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const;
};

/// Preconditioned iterative solver that relies on a stored LHS matrix
/** The template parameter nvars is the block size we want to use.
 * In a finite volume setting, the natural choice is the number of physical variables
 * or the number of PDEs in the system.
 */
class IterativeSolver : public IterativeSolverBase
{
protected:
	const SRMatrixView<a_real,a_int>& A;            ///< The LHS matrix context
	const Preconditioner<a_real,a_int>& prec;     ///< Preconditioner context

public:
	IterativeSolver(const SRMatrixView<a_real,a_int>& mat, const Preconditioner<a_real,a_int>& precond);

	/// Solves the linear system A x = b
	/** Note that usually, the two arguments cannot alias each other.
	 * \param[in] b  Right-hand side vector.
	 * \param [in|out] x Contains the solution in the same format as res on exit.
	 * \return Returns some metadata about the solver iterations performed
	 */
	virtual SolveInfo solve(const a_real *const b,
							a_real *const __restrict x) const = 0;
};

/// A solver that just applies the preconditioner repeatedly
class RichardsonSolver : public IterativeSolver
{
	using IterativeSolver::A;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeSolver::prec;

public:
	RichardsonSolver(const SRMatrixView<a_real,a_int>& mat,
					 const Preconditioner<a_real,a_int>& precond);

	SolveInfo solve(const a_real *const res, a_real *const __restrict du) const;
};

/// H.A. Van der Vorst's stabilized biconjugate gradient solver
/** Uses right-preconditioning only.
 */
class BiCGSTAB : public IterativeSolver
{
	using IterativeSolver::A;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeSolver::prec;

public:
	BiCGSTAB(const SRMatrixView<a_real,a_int>& mat, const Preconditioner<a_real,a_int>& precond);

	SolveInfo solve(const a_real *const res, a_real *const __restrict du) const;
};

/// Generalized Conjugate Residual solver
/** In exact arithmetic, this should be the same as a flexible GMRES.
 */
class GCR : public IterativeSolver
{
	using IterativeSolver::A;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeSolver::walltime;
	using IterativeSolver::cputime;
	using IterativeSolver::prec;
	int nrestart;

public:
	GCR(const SRMatrixView<a_real,a_int>& mat, const Preconditioner<a_real,a_int>& precond,
		int n_restart);

	SolveInfo solve(const a_real *const res, a_real *const __restrict du) const;
};

}
#endif
