/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include "testsolve.hpp"
#include "../src/blockmatrices.hpp"
#include "../src/coomatrix.hpp"

using namespace blasted;

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
template <int nvars>
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
template <int nvars>
class NoPrec : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	NoPrec(MatrixView<a_real,a_int> *const op) : Preconditioner<nvars>(op)
	{ }
	
	void compute()
	{ }
	
	void apply(const a_real *const r, 
			a_real *const z)
	{
#pragma omp parallel for simd default(shared)
		for(a_int i = 0; i < A->dim(); i++)
			z[i] = r[i];
	}
};

/// Jacobi preconditioner
template <int nvars>
class Jacobi : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	Jacobi(MatrixView<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }
	
	void compute() {
		A->precJacobiSetup();
	}

	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precJacobiApply(r, z);
	}
};

/// Symmetric Gauss-Seidel preconditioner
template <int nvars>
class SGS : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	SGS(MatrixView<a_real,a_int> *const op) : Preconditioner<nvars>(op) 
	{ }

	/// Sets D,L,U and inverts each D; also allocates temp storage
	void compute() {
		A->precSGSSetup();
	}

	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precSGSApply(r, z);
	}
};

/// ILU0 preconditioner
template <int nvars>
class ILU0 : public Preconditioner<nvars>
{
	using Preconditioner<nvars>::A;

public:
	ILU0(MatrixView<a_real,a_int> *const op) : Preconditioner<nvars>(op) { }

	/// Sets D,L,U and computes the ILU factorization
	void compute() {
		A->precILUSetup();
	}
	
	/// Solves Mz=r, where M is the preconditioner
	void apply(const a_real *const r, 
			a_real *const __restrict z) {
		A->precILUApply(r, z);
	}
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
	IterativeSolverBase() {
		resetRunTimes();
	}

	virtual ~IterativeSolverBase() { }
	
	/// Set tolerance and max iterations
	void setParams(const double toler, const int maxits) {
		maxiter = maxits; tol = toler;
	}
	
	/// Sets time accumulators to zero
	void resetRunTimes() {
		walltime = 0; cputime = 0;
	}

	/// Get timing data
	void getRunTimes(double& wall_time, double& cpu_time) const {
		wall_time = walltime; cpu_time = cputime;
	}
};

/// Preconditioned iterative solver that relies on a stored LHS matrix
/** The template parameter nvars is the block size we want to use.
 * In a finite volume setting, the natural choice is the number of physical variables
 * or the number of PDEs in the system.
 */
template <int nvars>
class IterativeSolver : public IterativeSolverBase
{
protected:
	MatrixView<a_real,a_int> *const A;        ///< The LHS matrix context

	/// Preconditioner context
	Preconditioner<nvars> *const prec;

public:
	IterativeSolver(MatrixView<a_real,a_int>* const mat, Preconditioner<nvars> *const precond)
		: A(mat), prec(precond)
	{ }

	/// Compute the preconditioner
	virtual void setupPreconditioner()
	{
		struct timeval time1, time2;
		gettimeofday(&time1, NULL);
		double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
		double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
		
		prec->compute();
		
		gettimeofday(&time2, NULL);
		double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
		double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
		walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	}

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
template <int nvars>
class RichardsonSolver : public IterativeSolver<nvars>
{
	using IterativeSolver<nvars>::A;
	using IterativeSolver<nvars>::maxiter;
	using IterativeSolver<nvars>::tol;
	using IterativeSolver<nvars>::walltime;
	using IterativeSolver<nvars>::cputime;
	using IterativeSolver<nvars>::prec;

public:
	RichardsonSolver(MatrixView<a_real,a_int>* const mat, Preconditioner<nvars> *const precond)
		: IterativeSolver<nvars>(mat, precond)
	{ }

	/** \param[in] res The right hand side vector
	 * \param[in] du The solution vector which is assumed to contain an initial solution
	 *
	 * \warning The two arguments must not alias each other.
	 */
	int solve(const a_real *const res, 
		a_real *const __restrict du) const
	{
		struct timeval time1, time2;
		gettimeofday(&time1, NULL);
		double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
		double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

		a_real resnorm = 100.0, bnorm = 0;
		int step = 0;
		const a_int N = A->dim();
		Vector<a_real> s(A->dim());
		Vector<a_real> ddu(A->dim());

		// norm of RHS
		bnorm = std::sqrt(dot(N, res, res));

		while(step < maxiter)
		{
			A->gemv3(-1.0,du, -1.0,res, s.data());

			resnorm = std::sqrt(dot(N, s.data(),s.data()));
			if(resnorm/bnorm < tol) break;

			prec->apply(s.data(), ddu.data());

			axpby(N, 1.0, du, 1.0, ddu.data());

			step++;
		}
		
		gettimeofday(&time2, NULL);
		double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
		double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
		walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
		std::cout << "  Final rel res norm = " << resnorm/bnorm << std::endl;
		return step;
	}
};

template<int bs>
int testSolveRichardson(const std::string precontype,
		const std::string mattype, const std::string storageorder, 
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps)
{
	std::cout << "Inputs: Prec = " << precontype << ", mat type = " << mattype
		<< ", tolerance = " << tol << " maxiter = " << maxiter
		<< ",\n Num build sweeps = " << nbuildswps << ", num apply sweeps = " << napplyswps << '\n';

	RawBSRMatrix<double,int> rm;
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);
	if(mattype == "csr")
		coom.convertToCSR(&rm);
	else
		if(storageorder == "rowmajor")
			coom.convertToBSR<bs,RowMajor>(&rm);
		else
			coom.convertToBSR<bs,ColMajor>(&rm);

	const double *const ans = readDenseMatrixMarket<double>(xfile);
	const double *const b = readDenseMatrixMarket<double>(bfile);
	double *const x = new double[rm.nbrows*bs];

	MatrixView<double,int>* mat = nullptr;
	if(mattype == "csr")
		mat = new CSRMatrixView<double,int>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);
	else
		if(storageorder == "rowmajor")
			mat = new BSRMatrixView<double,int,bs,RowMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);
		else
			mat = new BSRMatrixView<double,int,bs,ColMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);

	Preconditioner<bs>* prec = nullptr;
	if(precontype == "jacobi")
		prec = new Jacobi<bs>(mat);
	else if(precontype == "sgs")
		prec = new SGS<bs>(mat);
	else if(precontype == "ilu0")
		prec = new ILU0<bs>(mat);
	else {
		std::cout << " ! Invalid preconditioner option!\n";
		std::abort();
	}

	IterativeSolver<bs>* solver = new RichardsonSolver<bs>(mat,prec);

	solver->setupPreconditioner();
	solver->setParams(tol,maxiter);
	int iters = solver->solve(b, x);
	std::cout << " Num iters = " << iters << std::endl;

	for(int i = 0; i < rm.nbrows*bs; i++) {
		assert(std::fabs(x[i]-ans[i]) < 100*tol);
	}

	delete solver;
	delete prec;
	delete mat;

	delete [] rm.browptr;
	delete [] rm.bcolind;
	delete [] rm.vals;
	delete [] rm.diagind;
	delete [] x;
	delete [] b;
	delete [] ans;

	return 0;
}

template
int testSolveRichardson<3>(const std::string precontype,
		const std::string mattype, const std::string storageorder, 
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

template
int testSolveRichardson<7>(const std::string precontype,
		const std::string mattype, const std::string storageorder, 
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

