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

	/// Solves the linear system A du = -r
	/** Note that usually, the two arguments cannot alias each other.
	 * \param[in] res The residual vector stored as a 2D array of size nelem x nvars 
	 * (nelem x 4 for 2D Euler)
	 * \param [in|out] du Contains the solution in the same format as res on exit.
	 * \return Returns the number of solver iterations performed
	 */
	virtual int solve(const a_real *const res, 
	                  a_real *const __restrict du) const = 0;
};

/// A solver that just applies the preconditioner repeatedly
class RichardsonSolver : public IterativeSolver
{
	using IterativeSolver::A;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeSolver::walltime;
	using IterativeSolver::cputime;
	using IterativeSolver::prec;

public:
	RichardsonSolver(const SRMatrixView<a_real,a_int>& mat, const Preconditioner<a_real,a_int>& precond);

	/** \param[in] res The right hand side vector
	 * \param[in] du The solution vector which is assumed to contain an initial solution
	 *
	 * \warning The two arguments must not alias each other.
	 */
	int solve(const a_real *const res, a_real *const __restrict du) const;
};

/// H.A. Van der Vorst's stabilized biconjugate gradient solver
/** Uses right-preconditioning only.
 */
class BiCGSTAB : public IterativeSolver
{
	using IterativeSolver::A;
	using IterativeSolver::maxiter;
	using IterativeSolver::tol;
	using IterativeSolver::walltime;
	using IterativeSolver::cputime;
	using IterativeSolver::prec;

public:
	BiCGSTAB(const SRMatrixView<a_real,a_int>& mat, const Preconditioner<a_real,a_int>& precond);

	int solve(const a_real *const res, a_real *const __restrict du) const;
};


}
#endif
