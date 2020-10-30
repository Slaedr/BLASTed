/** \file
 * \brief Test for convergence of async triangular solve to the solution of exact triangular solve
 */

#include <algorithm>
#include "../../src/blas/blas1.hpp"
#include "../../src/kernels/kernels_ilu_apply.hpp"
#include "iter_prec_conv.hpp"

using namespace blasted;

template <int bs>
using Blk = Block_t<double,bs,ColMajor>;
template <int bs>
using Seg = Segment_t<double,bs>;

enum TriangleType { LOWER, UPPER };

template <int bs>
static device_vector<double> getExactLowerSolve(const CRawBSRMatrix<double,int>& mat,
                                                const device_vector<double>& iluvals,
                                                const device_vector<double>& rhs,
                                                const int thread_chunk_size);
template <int bs>
static device_vector<double> getExactUpperSolve(const CRawBSRMatrix<double,int>& mat,
                                                const device_vector<double>& iluvals,
                                                const device_vector<double>& rhs,
                                                const int thread_chunk_size);

// template <int bs>
// static void runPartiallyAsyncTest(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& iluvals,
//                                   const device_vector<double>& rhs,
//                                   const device_vector<double>& exactsoln, const TriangleType triangle,
//                                   const std::string initialization, const int thread_chunk_size,
//                                   const double tol, const int maxsweeps);

template <int bs>
static void runFullyAsyncTest(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& iluvals,
                              const device_vector<double>& rhs,
                              const device_vector<double>& exactsoln, const TriangleType triangle,
                              const std::string initialization, const int thread_chunk_size,
                              const double tol, const int maxsweeps);

template <int bs>
int test_async_triangular_solve(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                const double tol, const int maxiter, const bool usescale,
                                const int thread_chunk_size, const std::string initialization)
{
	int ierr = 0;

	//const device_vector<double> scale = getScalingVector<bs>(&mat);
	const device_vector<double> scale = usescale ? getScalingVector<bs>(&mat) : device_vector<double>(0);
	// const device_vector<double> scale(mat.nbrows*bs,1.0);
	if(usescale) {
		printf(" Computing factors of symmetrically scaled matrix.\n");
		assert(scale.size() > 0);
	} else {
		printf(" Computing factors of original (unscaled) matrix.\n");
		assert(scale.size() == 0);
	}
	const device_vector<double> iluvals = getExactILU<bs>(&mat, plist, scale);
	const device_vector<double> rhs(mat.nbrows*bs, 1.1);

	const device_vector<double> exact_low = getExactLowerSolve<bs>(mat, iluvals, rhs, thread_chunk_size);
	const device_vector<double> exact_up = getExactUpperSolve<bs>(mat, iluvals, rhs, thread_chunk_size);

	const double exactlownorm = maxnorm(exact_low);
	const double exactupnorm = maxnorm(exact_up);

	printf(" Norm of exact solutions: L = %g, U = %g.\n", exactlownorm, exactupnorm);

	printf(" Testing lower triangular solve..\n");
	runFullyAsyncTest<bs>(mat, iluvals, rhs, exact_low, LOWER, initialization, thread_chunk_size,
	                      tol, maxiter);
	printf("\n Testing upper triangular solve..\n");
	runFullyAsyncTest<bs>(mat, iluvals, rhs, exact_up, UPPER, initialization, thread_chunk_size,
	                      tol, maxiter);

	return ierr;
}

template
int test_async_triangular_solve<1>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                   const double tol, const int maxiter, const bool usescale,
                                   const int thread_chunk_size, const std::string initialization);
template
int test_async_triangular_solve<4>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                   const double tol, const int maxiter, const bool usescale,
                                   const int thread_chunk_size, const std::string initialization);
template
int test_async_triangular_solve<5>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                   const double tol, const int maxiter, const bool usescale,
                                   const int thread_chunk_size, const std::string initialization);
template
int test_async_triangular_solve<7>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                   const double tol, const int maxiter, const bool usescale,
                                   const int thread_chunk_size, const std::string initialization);

template <int bs>
static void lowerSolve(const CRawBSRMatrix<double,int>& mat,
                       const device_vector<double>& iluvals,
                       const device_vector<double>& rhs,
                       const bool usethreads, const int thread_chunk_size,
                       device_vector<double>& soln);
template <int bs>
static void upperSolve(const CRawBSRMatrix<double,int>& mat,
                       const device_vector<double>& iluvals,
                       const device_vector<double>& rhs,
                       const bool usethreads, const int thread_chunk_size,
                       device_vector<double>& soln);

static double maxnorm_diff(const device_vector<double>& x, const device_vector<double>& y);

template <int bs>
static void check_convergence(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& tsol,
                              const device_vector<double>& exactsoln, const double exactsolnorm,
                              const double tol, const std::string initialization,
                              int& isweep, int& curmaxsweeps, bool& converged);

template <int bs>
void runFullyAsyncTest(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& iluvals,
                       const device_vector<double>& rhs,
                       const device_vector<double>& exactsoln, const TriangleType triangle,
                       const std::string initialization, const int thread_chunk_size,
                       const double tol, const int maxsweeps)
{
	device_vector<double> tsol(mat.nbrows*bs);

	if(initialization == "exact")
		tsol = exactsoln;
	else if(initialization == "zero")
		std::fill(tsol.begin(), tsol.end(), 0.0);
	else
		throw std::runtime_error("Invalid initialization!");

	printf(" Initial error = %g.\n", maxnorm_diff(exactsoln, tsol));

	int isweep = 0;
	bool converged = (initialization == "exact") ? true : false;
	int curmaxsweeps = maxsweeps;

	const double exactsolnorm = maxnorm(exactsoln);

	printf(" %5s %10s\n", "Sweep", "Diff-norm");

	while(isweep < curmaxsweeps)
	{
		if(initialization == "exact")
			tsol = exactsoln;
		else if(initialization == "zero")
			std::fill(tsol.begin(), tsol.end(), 0.0);
		else
			throw std::runtime_error("Invalid initialization!");

		if(triangle == LOWER)
		{
			if(bs == 1)
			{
#pragma omp parallel default(shared)
				for(int subsweep = 0; subsweep <= isweep; subsweep++)
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
					for(int i = 0; i < mat.nbrows; i++)
					{
						tsol[i] = scalar_unit_lower_triangular(&iluvals[0], mat.bcolind, mat.browptr[i],
						                                       mat.diagind[i], rhs[i], &tsol[0]);
					}
			}
			else
			{
				const Blk<bs> *ilu = reinterpret_cast<const Blk<bs>*>(&iluvals[0]);
				const Seg<bs> *r = reinterpret_cast<const Seg<bs>*>(&rhs[0]);
				Seg<bs> *y = reinterpret_cast<Seg<bs>*>(&tsol[0]);

#pragma omp parallel default(shared)
				for(int subsweep = 0; subsweep <= isweep; subsweep++)
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
					for(int i = 0; i < mat.nbrows; i++)
					{
						block_unit_lower_triangular<double,int,bs,ColMajor>
							(ilu, mat.bcolind, mat.browptr[i], mat.diagind[i], r[i], i, y);
					}
			}
		}
		else
		{
			if(bs == 1) {
#pragma omp parallel default(shared)
				for(int subsweep = 0; subsweep <= isweep; subsweep++)
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
					for(int i = mat.nbrows-1; i >= 0; i--)
					{
						tsol[i] = scalar_upper_triangular(&iluvals[0], mat.bcolind, mat.diagind[i], 
						                                  mat.browptr[i+1], 1.0/iluvals[mat.diagind[i]],
						                                  rhs[i], &tsol[0]);
					}
			}
			else
			{
				const Blk<bs> *ilu = reinterpret_cast<const Blk<bs>*>(&iluvals[0]);
				const Seg<bs> *y = reinterpret_cast<const Seg<bs>*>(&rhs[0]);
				Seg<bs> *z = reinterpret_cast<Seg<bs>*>(&tsol[0]);

#pragma omp parallel default(shared)
				for(int subsweep = 0; subsweep <= isweep; subsweep++)
#pragma omp for schedule(dynamic, thread_chunk_size) nowait
					for(int i = mat.nbrows-1; i >= 0; i--)
					{
						block_upper_triangular<double,int,bs,ColMajor>
							(ilu, mat.bcolind, mat.diagind[i], mat.browptr[i+1], y[i], i, z);
					}
			}
		}

		check_convergence<bs>(mat, tsol, exactsoln, exactsolnorm, tol, initialization,
		                      isweep, curmaxsweeps, converged);
	}

	assert(converged);
}

template <int bs>
void runPartiallyAsyncTest(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& iluvals,
                           const device_vector<double>& rhs,
                           const device_vector<double>& exactsoln, const TriangleType triangle,
                           const std::string initialization, const int thread_chunk_size,
                           const double tol, const int maxsweeps)
{
	device_vector<double> tsol(mat.nbrows*bs);

	if(initialization == "exact")
		tsol = exactsoln;
	else if(initialization == "zero")
		std::fill(tsol.begin(), tsol.end(), 0.0);
	else
		throw std::runtime_error("Invalid initialization!");

	printf(" Initial error = %g.\n", maxnorm_diff(exactsoln, tsol));

	int isweep = 0;
	bool converged = (initialization == "exact") ? true : false;
	int curmaxsweeps = maxsweeps;

	const double exactsolnorm = maxnorm(exactsoln);

	printf(" %5s %10s\n", "Sweep", "Diff-norm");

	while(isweep < curmaxsweeps)
	{
		if(triangle == LOWER)
			lowerSolve<bs>(mat, iluvals, rhs, true, thread_chunk_size, tsol);
		else
			upperSolve<bs>(mat, iluvals, rhs, true, thread_chunk_size, tsol);

		check_convergence<bs>(mat, tsol, exactsoln, exactsolnorm, tol, initialization,
		                      isweep, curmaxsweeps, converged);
	}

	assert(converged);
}

template <int bs>
void check_convergence(const CRawBSRMatrix<double,int>& mat, const device_vector<double>& tsol,
                       const device_vector<double>& exactsoln, const double exactsolnorm,
                       const double tol, const std::string initialization,
                       int& isweep, int& curmaxsweeps, bool& converged)
{
	double errnorm;

	if(initialization == "exact") {
		// If initial L and U are exact, the initial error is zero. So don't normalize.
		errnorm = maxnorm_diff(tsol, exactsoln);
	}
	else {
		errnorm = maxnorm_diff(tsol, exactsoln)/exactsolnorm;
	}

	printf(" %5d %10.3g\n", isweep, errnorm); fflush(stdout);

	assert(std::isfinite(errnorm));

	isweep++;

	if(converged) {
		// The solution should not change
		assert(errnorm < tol);
	}
	else {
		// If tolerance is reached, see if the solution remains the same for 2 more iterations
		if(errnorm < tol) {
			converged = true;
			curmaxsweeps = isweep+2;
		}
	}
}

template <int bs>
void lowerSolve(const CRawBSRMatrix<double,int>& mat,
                const device_vector<double>& iluvals,
                const device_vector<double>& rhs,
                const bool usethreads, const int thread_chunk_size,
                device_vector<double>& soln)
{
	if(bs == 1)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(int i = 0; i < mat.nbrows; i++)
		{
			soln[i] = scalar_unit_lower_triangular(&iluvals[0], mat.bcolind, mat.browptr[i],
			                                        mat.diagind[i], rhs[i], &soln[0]);
		}
	else
	{
		const Blk<bs> *ilu = reinterpret_cast<const Blk<bs>*>(&iluvals[0]);
		const Seg<bs> *r = reinterpret_cast<const Seg<bs>*>(&rhs[0]);
		Seg<bs> *y = reinterpret_cast<Seg<bs>*>(&soln[0]);

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(int i = 0; i < mat.nbrows; i++)
		{
			block_unit_lower_triangular<double,int,bs,ColMajor>
				(ilu, mat.bcolind, mat.browptr[i], mat.diagind[i], r[i], i, y);
		}
	}
}

template <int bs>
void upperSolve(const CRawBSRMatrix<double,int>& mat,
                const device_vector<double>& iluvals,
                const device_vector<double>& rhs,
                const bool usethreads, const int thread_chunk_size,
                device_vector<double>& soln)
{
	if(bs == 1)
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(int i = mat.nbrows-1; i >= 0; i--)
		{
			soln[i] = scalar_upper_triangular(&iluvals[0], mat.bcolind, mat.diagind[i], 
			                                  mat.browptr[i+1], 1.0/iluvals[mat.diagind[i]],
			                                  rhs[i], &soln[0]);
		}
	else
	{
		const Blk<bs> *ilu = reinterpret_cast<const Blk<bs>*>(&iluvals[0]);
		const Seg<bs> *y = reinterpret_cast<const Seg<bs>*>(&rhs[0]);
		Seg<bs> *z = reinterpret_cast<Seg<bs>*>(&soln[0]);

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size) if(usethreads)
		for(int i = mat.nbrows-1; i >= 0; i--)
		{
			block_upper_triangular<double,int,bs,ColMajor>
				(ilu, mat.bcolind, mat.diagind[i], mat.browptr[i+1], y[i], i, z);
		}
	}
}

template <int bs>
device_vector<double> getExactLowerSolve(const CRawBSRMatrix<double,int>& mat,
                                         const device_vector<double>& iluvals,
                                         const device_vector<double>& rhs,
                                         const int thread_chunk_size)
{
	device_vector<double> ytemp(mat.nbrows*bs);
	lowerSolve<bs>(mat, iluvals, rhs, false, thread_chunk_size, ytemp);
	return ytemp;
}

template <int bs>
device_vector<double> getExactUpperSolve(const CRawBSRMatrix<double,int>& mat,
                                         const device_vector<double>& iluvals,
                                         const device_vector<double>& rhs,
                                         const int thread_chunk_size)
{
	device_vector<double> soln(mat.nbrows*bs);
	upperSolve<bs>(mat, iluvals, rhs, false, thread_chunk_size, soln);
	return soln;
}

double maxnorm_diff(const device_vector<double>& x, const device_vector<double>& y)
{
	assert(x.size() == y.size());

	double mn = 0;
#pragma omp parallel for default(shared) reduction(max:mn)
	for(size_t i = 0; i < x.size(); i++) {
		const double diff = std::abs(x[i]-y[i]);
		if(diff > mn)
			mn = diff;
	}

	return mn;
}
