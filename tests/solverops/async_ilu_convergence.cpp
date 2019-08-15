/** \file
 * \brief Implementation of test for convergence of async (B)ILU factorization with sweeps
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#undef NDEBUG

#include <stdexcept>
#include <string>
#include "utils/cmdoptions.hpp"
#include "../../src/kernels/kernels_ilu0_factorize.hpp"
#include "../../src/async_blockilu_factor.hpp"
#include "../../src/async_ilu_factor.hpp"
#include "../testutils.h"
#include "../testutils.hpp"
#include "iter_prec_conv.hpp"

using namespace blasted;

template <int bs>
using Blk = Block_t<double,bs,ColMajor>;

/// Computes norm of flattened vector of the difference between the unit lower triangular parts
template <int bs>
static double maxnorm_lower(const CRawBSRMatrix<double,int> *const mat,
                            const device_vector<double>& x, const device_vector<double>& y);

/// Computes norm of flattened vector of the difference between the upper triangular parts
template <int bs>
static double maxnorm_upper(const CRawBSRMatrix<double,int> *const mat,
                            const device_vector<double>& x, const device_vector<double>& y);

/// Carry out some checks
template <int bs>
static void check_initial(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                          const device_vector<double>& scale, const int thread_chunk_size,
                          const std::string initialization,
                          const device_vector<double>& exactilu, const device_vector<double>& iluvals,
                          const double initLerr, const double initUerr);

template <int bs>
int test_ilu_convergence(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                         const double tol, const int maxsweeps,
                         const int thread_chunk_size, const std::string initialization)
{
	int ierr = 0;

	const device_vector<double> scale = (bs==1) ? getScalingVector(&mat) : device_vector<double>(0);

	const device_vector<double> exactilu = getExactILU<bs>(&mat,plist,scale);

	device_vector<double> iluvals(mat.nnzb*bs*bs);

	if(initialization == "orig")
		// Initialize with original matrix
		for(int i = 0; i < mat.nnzb*bs*bs; i++)
			iluvals[i] = mat.vals[i];
	else if(initialization == "exact")
		for(int i = 0; i < mat.nnzb*bs*bs; i++)
			iluvals[i] = exactilu[i];
	else
		throw std::runtime_error("Unsupported initialization requested!");

	const double initLerr = maxnorm_lower<bs>(&mat, iluvals, exactilu);
	const double initUerr = maxnorm_upper<bs>(&mat, iluvals, exactilu);

	check_initial<bs>(mat, plist, scale, thread_chunk_size, initialization, exactilu, iluvals,
	                  initLerr, initUerr);

	Blk<bs> *const ilu = reinterpret_cast<Blk<bs>*>(&iluvals[0]);
	const Blk<bs> *const mvals = reinterpret_cast<const Blk<bs>*>(mat.vals);

	int isweep = 0;
	double Lerr = 1.0, Uerr = 1.0, ilures = 1.0;
	bool converged = (initialization == "exact") ? true : false;
	int curmaxsweeps = maxsweeps;

	printf(" %5s %10s %10s %10s\n", "Sweep", "L-norm","U-norm","NL-Res");

	while(isweep < curmaxsweeps)
	{
		if(bs == 1)
		{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
			for(int irow = 0; irow < mat.nbrows; irow++)
			{
				async_ilu0_factorize_kernel<double,int,true,true>(&mat, plist, irow,
				                                                  &scale[0], &scale[0], &iluvals[0]);
			}
		}
		else
		{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
			for(int irow = 0; irow < mat.nbrows; irow++)
			{
				async_block_ilu0_factorize<double,int,bs,ColMajor>(&mat, mvals, plist, irow, ilu);
			}
		}

		if(initialization == "exact") {
			// If initial L and U are exact, the initial error is zero. So don't normalize.
			Lerr = maxnorm_lower<bs>(&mat, iluvals, exactilu);
			Uerr = maxnorm_upper<bs>(&mat, iluvals, exactilu);
		}
		else {
			Lerr = maxnorm_lower<bs>(&mat, iluvals, exactilu)/initLerr;
			Uerr = maxnorm_upper<bs>(&mat, iluvals, exactilu)/initUerr;
		}

		ilures = (bs == 1) ?
			scalar_ilu0_nonlinear_res<double,int,true,true>(&mat, plist, thread_chunk_size, &scale[0],
			                                                &scale[0], &iluvals[0])
			:
			block_ilu0_nonlinear_res<double,int,bs,ColMajor>(&mat, plist, &iluvals[0], thread_chunk_size);

		printf(" %5d %10.3g %10.3g %10.3g\n", isweep, Lerr, Uerr, ilures); fflush(stdout);

		assert(std::isfinite(Lerr));
		assert(std::isfinite(Uerr));
		assert(std::isfinite(ilures));

		isweep++;

		if(converged) {
			// The solution should not change
			assert(Lerr < tol);
			assert(Uerr < tol);
		}
		else {
			// If tolerance is reached, see if the solution remains the same for 2 more iterations
			if(Lerr < tol && Uerr < tol) {
				converged = true;
				curmaxsweeps = isweep+2;
			}
		}
	}

	assert(converged);

	return ierr;
}

template
int test_ilu_convergence<1>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                            const double tol, const int maxsweeps,
                            const int thread_chunk_size, const std::string initialization);
template
int test_ilu_convergence<4>(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                            const double tol, const int maxsweeps,
                            const int thread_chunk_size, const std::string initialization);

template <int bs>
double maxnorm_lower(const CRawBSRMatrix<double,int> *const mat,
                     const device_vector<double>& x, const device_vector<double>& y)
{
	constexpr int bs2 = bs*bs;
	double maxnorm = 0;

#pragma omp parallel for default(shared) reduction(max:maxnorm)
	for(int irow = 0; irow < mat->nbrows; irow++)
	{
		for(int jj = mat->browptr[irow]; jj < mat->diagind[irow]; jj++)
		{
			double blknorm = 0;
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
				{
					const double diff = std::abs(x[jj*bs2 + i*bs+j] - y[jj*bs2 + i*bs+j]);
					if (blknorm < diff)
						blknorm = diff;
				}
			if (maxnorm < blknorm)
				maxnorm = blknorm;
		}
	}

	return maxnorm;
}

template <int bs>
double maxnorm_upper(const CRawBSRMatrix<double,int> *const mat,
                     const device_vector<double>& x, const device_vector<double>& y)
{
	constexpr int bs2 = bs*bs;
	double maxnorm = 0;

#pragma omp parallel for default(shared) reduction(max:maxnorm)
	for(int irow = 0; irow < mat->nbrows; irow++)
	{
		for(int jj = mat->diagind[irow]; jj < mat->browendptr[irow]; jj++)
		{
			double blknorm = 0;
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
				{
					const double diff = std::abs(x[jj*bs2 + i*bs+j] - y[jj*bs2 + i*bs+j]);
					if (blknorm < diff)
						blknorm = diff;
				}

			if (maxnorm < blknorm)
				maxnorm = blknorm;
		}
	}

	return maxnorm;
}

namespace blasted {

template <int bs>
device_vector<double> getExactILU(const CRawBSRMatrix<double,int> *const mat,
                                  const ILUPositions<int>& plist, const device_vector<double>& scale)
{
	device_vector<double> iluvals(mat->nnzb*bs*bs);

	Blk<bs> *const ilu = reinterpret_cast<Blk<bs>*>(&iluvals[0]);
	const Blk<bs> *const mvals = reinterpret_cast<const Blk<bs>*>(mat->vals);

	for(int irow = 0; irow < mat->nbrows; irow++)
	{
		if(bs == 1)
			async_ilu0_factorize_kernel<double,int,true,true>(mat, plist, irow,
			                                                  &scale[0], &scale[0], &iluvals[0]);
		else
			async_block_ilu0_factorize<double,int,bs,ColMajor>(mat, mvals, plist, irow, ilu);
	}

	return iluvals;
}

template
device_vector<double> getExactILU<1>(const CRawBSRMatrix<double,int> *const mat,
                                     const ILUPositions<int>& plist, const device_vector<double>& scale);
template
device_vector<double> getExactILU<4>(const CRawBSRMatrix<double,int> *const mat,
                                     const ILUPositions<int>& plist, const device_vector<double>& scale);

device_vector<double> getScalingVector(const CRawBSRMatrix<double,int> *const mat)
{
	device_vector<double> scale(mat->nbrows);

#pragma omp parallel for simd default(shared)
	for(int i = 0; i < mat->nbrows; i++)
		scale[i] = 1.0/std::sqrt(mat->vals[mat->diagind[i]]);

	return scale;
}

}

template <int bs>
void check_initial(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                   const device_vector<double>& scale, const int thread_chunk_size,
                   const std::string initialization,
                   const device_vector<double>& exactilu, const device_vector<double>& iluvals,
                   const double initLerr, const double initUerr)
{
	// check maxnorm functions
	const double checkLerr = maxnorm_lower<bs>(&mat, exactilu, exactilu);
	printf(" Error of exact L factor = %5.5g.\n", checkLerr);
	assert(checkLerr < 1e-15);
	const double checkUerr = maxnorm_upper<bs>(&mat, exactilu, exactilu);
	printf(" Error of exact U factor = %5.5g.\n", checkUerr);
	assert(checkUerr < 1e-15);

	const double initilures = (bs == 1) ?
		scalar_ilu0_nonlinear_res<double,int,true,true>(&mat, plist, thread_chunk_size, &scale[0],
		                                                &scale[0], &iluvals[0])
		:
		block_ilu0_nonlinear_res<double,int,bs,ColMajor>(&mat, plist, &iluvals[0], thread_chunk_size);

	printf(" Initial lower and upper errors = %f, %f\n", initLerr, initUerr);
	printf(" Initial nonlinear ILU residual = %f.\n", initilures);

	// check ilu remainder
	const double checkilures = (bs == 1) ?
		scalar_ilu0_nonlinear_res<double,int,true,true>(&mat, plist, thread_chunk_size, &scale[0],
		                                                &scale[0], &exactilu[0])
		:
		block_ilu0_nonlinear_res<double,int,bs,ColMajor>(&mat, plist, &exactilu[0], thread_chunk_size);
	printf(" ILU remainder of serial factorization = %g, rel. residual = %g.\n", checkilures,
	       checkilures/initilures);
	fflush(stdout);
	if(initialization != "exact")
		assert(checkilures/initilures < 2.2e-16);
}

