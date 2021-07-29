/** \file solvers.cpp
 * \brief Some solver routines
 * \author Aditya Kashi
 * \date 2017-11
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "device_container.hpp"
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

inline void vecassign(const a_int N, const a_real *const __restrict__ a,
					  a_real *const __restrict__ b)
{
#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		b[i] = a[i];
	}
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

IterativeSolver::IterativeSolver(const SRMatrixView<a_real,a_int>& mat,
                                 const Preconditioner<a_real,a_int>& precond)
	: A(mat), prec(precond)
{ }

RichardsonSolver::RichardsonSolver(const SRMatrixView<a_real,a_int>& mat,
                                   const Preconditioner<a_real,a_int>& precond)
	: IterativeSolver(mat, precond)
{ }

SolveInfo RichardsonSolver::solve(const a_real *const res, a_real *const __restrict du) const
{
	SolveInfo info;
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real resnorm=0;
	int step = 0;
	const a_int N = A.dim();
	device_vector<a_real> s(A.dim());
	device_vector<a_real> ddu(A.dim());

	// norm of RHS
	const a_real bnorm = std::sqrt(dot(N, res, res));

	while(step < maxiter)
	{
		A.gemv3(-1.0,du, 1.0,res, s.data());

		resnorm = std::sqrt(dot(N, s.data(),s.data()));
		if(step % 10 == 0)
			std::cout << "  Step " << step << ", Rel res norm = " << resnorm/bnorm << std::endl;
		if(resnorm/bnorm < tol) break;

		prec.apply(s.data(), ddu.data());

		axpby(N, 1.0, du, 1.0, ddu.data());

		step++;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	info.walltime += (finalwtime-initialwtime);
	info.cputime += (finalctime-initialctime);
	info.iters = step;
	info.resnorm = resnorm;
	info.bnorm = bnorm;
	std::cout << "  Final rel res norm = " << resnorm/bnorm << std::endl;
	return info;
}

BiCGSTAB::BiCGSTAB(const SRMatrixView<a_real,a_int>& mat,
				   const Preconditioner<a_real,a_int>& precond)
	: IterativeSolver(mat, precond)
{ }

SolveInfo BiCGSTAB::solve(const a_real *const rhs, a_real *const __restrict xsol) const
{
	a_real resnorm = 100.0, bnorm = 0;
	int step = 0;
	const a_int N = A.dim();

	SolveInfo info;

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	a_real omega = 1.0, rho, rhoold = 1.0, alpha = 1.0, beta;
	device_vector<a_real> rhat(N);
	device_vector<a_real> r(N);
	device_vector<a_real> p(N);
	device_vector<a_real> v(N);
	device_vector<a_real> y(N);
	device_vector<a_real> z(N);
	device_vector<a_real> t(N);
	device_vector<a_real> g(N);

#pragma omp parallel for simd default(shared)
	for(a_int i = 0; i < N; i++) {
		p[i] = 0;
		v[i] = 0;
	}

	// r := rhs - A xsol
	A.gemv3(-1.0,xsol, 1.0,rhs, r.data());

	// norm of RHS
#pragma omp parallel for simd reduction(+:bnorm) default(shared)
	for(a_int iel = 0; iel < N; iel++)
	{
		bnorm += rhs[iel]*rhs[iel];
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

		// xsol <- xsol + alpha y + omega z
		axpbypcz(N, 1.0,xsol, alpha,y.data(), omega,z.data());

		// r <- r - omega t
		axpby(N, 1.0,r.data(), -omega,t.data());

		// check convergence or `lucky' breakdown
		resnorm = std::sqrt( dot(N, r.data(), r.data()) );
		if(step % 10 == 0)
			std::cout << "    Step " << step << ", Rel res norm = " << resnorm/bnorm << std::endl;

		if(resnorm/bnorm < tol) break;

		rhoold = rho;
		step++;
	}

	/*if(step == maxiter)
		std::cout << " ! BiCGSTAB: Hit max iterations!\n";*/
	std::cout << "  Final rel res norm = " << resnorm/bnorm << std::endl;
	info.iters = step+1;
	info.resnorm = resnorm;
	info.bnorm = bnorm;
	
	gettimeofday(&time2, NULL);
	const double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	const double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	info.walltime += (finalwtime-initialwtime);
	info.cputime += (finalctime-initialctime);
	return info;
}


GCR::GCR(const SRMatrixView<a_real,a_int>& mat,
		 const Preconditioner<a_real,a_int>& precond, const int n_restart)
	: IterativeSolver(mat, precond), nrestart{n_restart}
{ }

SolveInfo GCR::solve(const a_real *const b, a_real *const __restrict x) const
{
	SolveInfo info;
	assert(info.precapplywtime == 0);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	const a_int N = A.dim();

	device_vector<a_real> res(N);
	device_vector<a_real> z(N);

	device_vector<a_real> beta(nrestart+1);
	std::vector<device_vector<a_real>> p(nrestart), q(nrestart);
	for(int i = 0; i < nrestart; i++) {
		p[i].resize(N);
		q[i].resize(N);
	}

	// norm of RHS
	const a_real bnorm = sqrt(dot(N, b, b));
	std::cout << "   GCR: RHS norm = " << bnorm << std::endl;

	a_real resnorm = 1.0;
	int step = 0;

	while(step < maxiter)
	{
		// r := b - A x
		A.gemv3(-1.0,x, 1.0,b, res.data());
		resnorm = dot(N, res.data(), res.data());

		prec.apply(res.data(), p[0].data());

		A.apply(p[0].data(), q[0].data());

		for(int k = 0; k < nrestart; k++)
		{
			const a_real alpha = dot(N, res.data() ,q[k].data())
				/ dot(N, q[k].data() ,q[k].data() );
			// x <- x + alpha p
			axpby(N, 1.0,x, alpha, p[k].data());
			// res <- res - alpha q
			axpby(N, 1.0,res.data(), -alpha, q[k].data());

			//resnorm = norm_vector_l2(res);
			resnorm = sqrt(dot(N, res.data(), res.data()));
			if(step % 10 == 0) {
				printf("      Step %d: Rel res = %g\n", step, resnorm/bnorm);
				fflush(stdout);
			}
			step++;

			if(resnorm/bnorm < tol)
				break;
			if(k == nrestart-1)
				break;
			if(step >= maxiter)
				break;

			prec.apply(res.data(), z.data());

			A.apply(z.data(), q[k+1].data());
			vecassign(N, z.data(), p[k+1].data());

			// optimize this witha multi-dot
			for(int i = 0; i < k+1; i++) {
				beta[i] = -dot(N, q[k+1].data(), q[i].data()) / dot(N, q[i].data(), q[i].data());
			}

			for(int l = 0; l < k+1; l++) {
#pragma omp parallel for simd default(shared)
				for(int i = 0; i < N; i++) {
					p[k+1][i] += beta[l] * p[l][i];
					q[k+1][i] += beta[l] * q[l][i];
				}
			}
		}

		if(resnorm/bnorm < tol) {
			break;
		}
	}

	info.converged = resnorm/bnorm <= tol ? true : false;
	info.iters = step;
	info.resnorm = resnorm;
	info.bnorm = bnorm;

	std::cout << "  Final rel res norm = " << resnorm/bnorm << std::endl;

	gettimeofday(&time2, NULL);
	const double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	const double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	info.walltime += (finalwtime-initialwtime);
	info.cputime += (finalctime-initialctime);
	return info;
}


}
