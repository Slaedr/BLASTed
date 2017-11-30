/** \file solvers.cpp
 * \brief Some solver routines
 * \author Aditya Kashi
 * \date 2017-11
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "solvers.hpp"

namespace blasted {
	
NoPrec::NoPrec(MatrixView<a_real,a_int> *const op) : Preconditioner(op)
{ }

void NoPrec::compute() { }

void NoPrec::apply(const a_real *const r, a_real *const __restrict z)
{
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < A->dim(); i++)
		z[i] = r[i];
}
	
Jacobi::Jacobi(MatrixView<a_real,a_int> *const op) : Preconditioner(op) { }

void Jacobi::compute() {
	A->precJacobiSetup();
}

void Jacobi::apply(const a_real *const r, a_real *const __restrict z) {
	A->precJacobiApply(r, z);
}

SGS::SGS(MatrixView<a_real,a_int> *const op) : Preconditioner(op) 
{ }

void SGS::compute() {
	A->precSGSSetup();
}

void SGS::apply(const a_real *const r, a_real *const __restrict z) {
	A->precSGSApply(r, z);
}

ILU0::ILU0(MatrixView<a_real,a_int> *const op) : Preconditioner(op) { }

/// Sets D,L,U and computes the ILU factorization
void ILU0::compute() {
	A->precILUSetup();
}

/// Solves Mz=r, where M is the preconditioner
void ILU0::apply(const a_real *const r, a_real *const __restrict z) {
	A->precILUApply(r, z);
}

IterativeSolverBase::IterativeSolverBase() {
	resetRunTimes();
}

IterativeSolverBase::~IterativeSolverBase() { }

void IterativeSolverBase::setParams(const double toler, const int maxits) {
	maxiter = maxits; tol = toler;
}

void IterativeSolverBase::resetRunTimes() {
	walltime = 0; cputime = 0;
}

void IterativeSolverBase::getRunTimes(double& wall_time, double& cpu_time) const {
	wall_time = walltime; cpu_time = cputime;
}

IterativeSolver::IterativeSolver(MatrixView<a_real,a_int>* const mat, Preconditioner *const precond)
	: A(mat), prec(precond)
{ }

void IterativeSolver::setupPreconditioner()
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

RichardsonSolver::RichardsonSolver(MatrixView<a_real,a_int>* const mat, Preconditioner *const precond)
	: IterativeSolver(mat, precond)
{ }

/** \param[in] res The right hand side vector
 * \param[in] du The solution vector which is assumed to contain an initial solution
 *
 * \warning The two arguments must not alias each other.
 */
int RichardsonSolver::solve(const a_real *const res, a_real *const __restrict du) const
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm;
	int step = 0;
	const a_int N = A->dim();
	std::vector<a_real> s(A->dim());
	std::vector<a_real> ddu(A->dim());

	// norm of RHS
	const a_real bnorm = std::sqrt(dot(N, res, res));

	while(step < maxiter)
	{
		A->gemv3(-1.0,du, -1.0,res, s.data());

		resnorm = std::sqrt(dot(N, s.data(),s.data()));
		std::cout << "  Rel res norm = " << resnorm/bnorm << std::endl;
		if(resnorm/bnorm < tol && step >= 5) break;

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

}
