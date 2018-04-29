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

#if 0
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
#endif

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

IterativeSolver::IterativeSolver(MatrixView<a_real,a_int>& mat, Preconditioner<a_real,a_int>& precond)
	: A(mat), prec(precond)
{ }

void IterativeSolver::setupPreconditioner()
{
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	prec.compute();
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
}

RichardsonSolver::RichardsonSolver(MatrixView<a_real,a_int>& mat, Preconditioner<a_real,a_int>& precond)
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

	a_real resnorm=0;
	int step = 0;
	const a_int N = A.dim();
	std::vector<a_real> s(A.dim());
	std::vector<a_real> ddu(A.dim());

	// norm of RHS
	const a_real bnorm = std::sqrt(dot(N, res, res));

	while(step < maxiter)
	{
		A.gemv3(-1.0,du, 1.0,res, s.data());

		resnorm = std::sqrt(dot(N, s.data(),s.data()));
		//std::cout << "Rel res norm = " << resnorm/bnorm << std::endl;
		if(resnorm/bnorm < tol) break;

		prec.apply(s.data(), ddu.data());

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

BiCGSTAB::BiCGSTAB(MatrixView<a_real,a_int>& mat, Preconditioner<a_real,a_int>& precond)
	: IterativeSolver(mat, precond)
{ }

int BiCGSTAB::solve(const a_real *const res, a_real *const __restrict du) const
{
	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	const a_int N = A.dim();

	a_real omega = 1.0, rho, rhoold = 1.0, alpha = 1.0, beta;
	std::vector<a_real> rhat(N);
	std::vector<a_real> r(N);
	std::vector<a_real> p(N);
	std::vector<a_real> v(N);
	std::vector<a_real> y(N);
	std::vector<a_real> z(N);
	std::vector<a_real> t(N);
	std::vector<a_real> g(N);

#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		p[i] = 0;
		v[i] = 0;
	}

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	// r := res - A du
	A.gemv3(-1.0,du, 1.0,res, r.data());

	// norm of RHS
#pragma omp parallel for simd reduction(+:bnorm) default(shared)
	for(a_int iel = 0; iel < N; iel++)
	{
		bnorm += res[iel];
		rhat[iel] = r[iel];
	}
	bnorm = std::sqrt(bnorm);
	std::cout << "   BiCGSTAB: RHS norm = " << bnorm << std::endl;

	while(step < maxiter)
	{
		// rho := rhat . r
		rho = dot(N, rhat.data(), r.data());
		beta = rho*alpha/(rhoold*omega);
		
		// p <- r + beta p - beta omega v
		axpbypcz(N, beta,p.data(), 1.0,r.data(), -beta*omega,v.data());
		
		// y <- Minv p
		prec.apply(p.data(), y.data());
		
		// v <- A y
		A.apply(y.data(), v.data());

		alpha = rho/dot(N, rhat.data(),v.data());

		// s <- r - alpha v, but reuse storage of r
		axpby(N, 1.0,r.data(), -alpha,v.data());

		// z <- Minv s
		prec.apply(r.data(), z.data());
		
		// t <- A z
		A.apply(z.data(), t.data());

		// For the left-preconditioned variant: g <- Minv t
		//prec->apply(t.data(),g.data());
		//omega = dot(N,g.data(),z.data())/dot(N,g.data(),g.data());

		omega = dot(N, t.data(),r.data()) / dot(N, t.data(),t.data());

		// du <- du + alpha y + omega z
		axpbypcz(N, 1.0,du, alpha,y.data(), omega,z.data());

		// r <- r - omega t
		axpby(N, 1.0,r.data(), -omega,t.data());

		// check convergence or `lucky' breakdown
		resnorm = std::sqrt( dot(N, r.data(), r.data()) );
		//std::cout << "Rel res norm = " << resnorm/bnorm << std::endl;

		if(resnorm/bnorm < tol) break;

		rhoold = rho;
		step++;
	}

	/*if(step == maxiter)
		std::cout << " ! BiCGSTAB: Hit max iterations!\n";*/
	std::cout << "  Final rel res norm = " << resnorm/bnorm << std::endl;
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	walltime += (finalwtime-initialwtime); cputime += (finalctime-initialctime);
	return step+1;
}


}
