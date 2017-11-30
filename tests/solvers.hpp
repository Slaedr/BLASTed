/** \file solvers.hpp
 * \brief Definitions of some solver routines
 * \author Aditya Kashi
 * \date 2017-11
 */

#ifndef BLASTED_SOLVERS_H
#define BLASTED_SOLVERS_H

#include "../src/linearoperator.hpp"

namespace blasted {

typedef int a_int;
typedef double a_real;

/// Vector or matrix addition
/** z <- pz + qx.
 * \param[in] N The length of the vectors
 */
inline void axpby(const a_int N, const a_real p, a_real *const __restrict z, 
	const a_real q, const a_real *const x)
{
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		z[i] = p*z[i] + q*x[i];
	}
}

/** z <- pz + qx + ry for vectors and matrices
 */
inline void axpbypcz(const a_int N, const a_real p, a_real *const z, 
	const a_real q, const a_real *const x,
	const a_real r, const a_real *const y)
{
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		z[i] = p*z[i] + q*x[i] + r*y[i];
	}
}

/// Dot product of vectors or `double dot' product of matrices
inline a_real dot(const a_int N, const a_real *const a, 
	const a_real *const b)
{
	a_real sum = 0;
#pragma omp parallel for simd default(shared) reduction(+:sum)
	for(a_int i = 0; i < N; i++)
		sum += a[i]*b[i];

	return sum;
}

/// Preconditioner, ie, performs one iteration to solve M z = r
/** Note that subclasses do not directly perform any computation but
 * delegate all computation to the relevant subclass of AbstractMatrix. 
 * As such, the precise preconditioning operation applied depends on 
 * which kind of matrix the LHS is stored as.
 */
class Preconditioner
{
protected:
	MatrixView<a_real,a_int>* A;

public:
	Preconditioner(MatrixView<a_real,a_int> *const op)
		: A(op)
	{ }
	
	virtual ~Preconditioner()
	{ }
	
	/// Computes the preconditioning matrix M
	virtual void compute() = 0;

	/// Applies the preconditioner Mz=r
	/** \param[in] r The right hand side vector
	 * \param [in|out] z Contains the solution
	 */
	virtual void apply(const a_real *const r, 
			a_real *const z) = 0;
};

/// Do-nothing preconditioner
/** The preconditioner is the identity matrix.
 */
class NoPrec : public Preconditioner
{
	using Preconditioner::A;

public:
	NoPrec(MatrixView<a_real,a_int> *const op);
	
	void compute();
	
	void apply(const a_real *const r, a_real *const z);
};

/// Jacobi preconditioner
class Jacobi : public Preconditioner
{
	using Preconditioner::A;

public:
	Jacobi(MatrixView<a_real,a_int> *const op);
	
	void compute();

	void apply(const a_real *const r, a_real *const __restrict z); 
};

/// Symmetric Gauss-Seidel preconditioner
class SGS : public Preconditioner
{
	using Preconditioner::A;

public:
	SGS(MatrixView<a_real,a_int> *const op);

	/// Sets D,L,U and inverts each D; also allocates temp storage
	void compute();

	void apply(const a_real *const r, a_real *const __restrict z);
};

/// ILU0 preconditioner
class ILU0 : public Preconditioner
{
	using Preconditioner::A;

public:
	ILU0(MatrixView<a_real,a_int> *const op);

	/// Sets D,L,U and computes the ILU factorization
	void compute();
	
	/// Solves Mz=r, where M is the preconditioner
	void apply(const a_real *const r, a_real *const __restrict z);
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
	MatrixView<a_real,a_int> *const A;        ///< The LHS matrix context
	Preconditioner *const prec;               ///< Preconditioner context

public:
	IterativeSolver(MatrixView<a_real,a_int>* const mat, Preconditioner *const precond);

	/// Compute the preconditioner
	virtual void setupPreconditioner();

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
	RichardsonSolver(MatrixView<a_real,a_int>* const mat, Preconditioner *const precond);

	/** \param[in] res The right hand side vector
	 * \param[in] du The solution vector which is assumed to contain an initial solution
	 *
	 * \warning The two arguments must not alias each other.
	 */
	int solve(const a_real *const res, a_real *const __restrict du) const;
};


}
#endif
