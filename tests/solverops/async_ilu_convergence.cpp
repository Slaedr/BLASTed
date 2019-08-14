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
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils/cmdoptions.hpp"
#include "utils/mpiutils.hpp"
#include "device_container.hpp"
#include "../../src/kernels/kernels_ilu0_factorize.hpp"
#include "../../src/async_blockilu_factor.hpp"
#include "../../src/async_ilu_factor.hpp"
#include "../testutils.h"
#include "../testutils.hpp"
#include "../poisson3d-fd/poisson_setup.h"

using namespace blasted;

DiscreteLinearProblem generateDiscreteProblem(const int argc, char *argv[]);

template <int bs>
int test_ilu_convergence(const DiscreteLinearProblem& dlp, const double tol, const int maxsweeps);

static int getBlockSize(const Mat A);

int main(int argc, char *argv[])
{
	if(argc < 2) {
		printf(" ! Please provide 'poisson' or 'file'\n");
		exit(-1);
	}

	PetscInitialize(&argc, &argv, NULL, NULL);
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	DiscreteLinearProblem dlp = generateDiscreteProblem(argc, argv);

	int ierr = 0;
	const int bs = getBlockSize(dlp.lhs);
	printf(" Input matrix: block size = %d\n", bs);

	const int maxsweeps = parsePetscCmd_int("-max_sweeps");
	const double tol = parseOptionalPetscCmd_real("-tolerance", 1e-25);

	switch(bs) {
	case 1:
		ierr = test_ilu_convergence<1>(dlp, tol, maxsweeps);
		break;
	case 4:
		ierr = test_ilu_convergence<4>(dlp, tol, maxsweeps);
		break;
	default:
		throw std::out_of_range("Block size " + std::to_string(bs) + " not supported!");
	}

	ierr = destroyDiscreteLinearProblem(&dlp);
	ierr = PetscFinalize();
	return ierr;
}

DiscreteLinearProblem generateDiscreteProblem(const int argc, char *argv[])
{
	DiscreteLinearProblem dlp;
	if(!strcmp(argv[1],"poisson"))
	{
		if(argc < 3) {
			printf(" ! Please provide a Poisson control file!\n");
			exit(-1);
		}

		dlp = setup_poisson_problem(argv[2]);
	}
	else {
		if(argc < 5) {
			printf(" ! Please provide filenames for LHS, RHS vector and exact solution (in order).\n");
			exit(-1);
		}

		int ierr = readLinearSystemFromFiles(argv[2], argv[3], argv[4], &dlp);
		assert(ierr == 0);
	}

	return dlp;
}

int getBlockSize(const Mat A)
{
	int bs = 0;
	int ierr = MatGetBlockSize(A, &bs);
	if(ierr != 0)
		throw Petsc_exception(ierr);

	// Check matrix type and adjust block size
	const char *mattype;
	ierr = MatGetType(A, &mattype);
	if(ierr != 0)
		throw Petsc_exception(ierr);
	if(!strcmp(mattype, MATSEQAIJ) || !strcmp(mattype, MATSEQAIJMKL) ||
	   !strcmp(mattype, MATMPIAIJ) || !strcmp(mattype, MATMPIAIJMKL) )
	{
		printf(" Matrix is a scalar SR type.\n");
		bs = 1;
	}
	else if(!strcmp(mattype, MATSEQBAIJ) || !strcmp(mattype, MATSEQBAIJMKL) ||
	        !strcmp(mattype, MATMPIBAIJ) || !strcmp(mattype, MATMPIBAIJMKL) )
		printf(" Matrix is a block SR type.\n");
	else
		throw std::runtime_error("Unsupported matrix type " + std::string(mattype));
	return bs;
}

template <int bs>
using Blk = Block_t<double,bs,ColMajor>;

template <int bs>
static device_vector<double> getExactILU(const CRawBSRMatrix<double,int> *const mat,
                                         const ILUPositions<int>& plist,
                                         const device_vector<double>& scale);

/// Computes norm of flattened vector of the difference between the unit lower triangular parts
template <int bs>
static double maxnorm_lower(const CRawBSRMatrix<double,int> *const mat,
                            const device_vector<double>& x, const device_vector<double>& y);

/// Computes norm of flattened vector of the difference between the upper triangular parts
template <int bs>
static double maxnorm_upper(const CRawBSRMatrix<double,int> *const mat,
                            const device_vector<double>& x, const device_vector<double>& y);

/// Computes symmetric scaling vector for scalar async ILU
static device_vector<double> getScalingVector(const CRawBSRMatrix<double,int> *const mat);

/// Carry out some checks
template <int bs>
static void check_initial(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                          const device_vector<double>& scale, const int thread_chunk_size,
                          const std::string initialization,
                          const device_vector<double>& exactilu, const device_vector<double>& iluvals,
                          const double initLerr, const double initUerr);

template <int bs>
int test_ilu_convergence(const DiscreteLinearProblem& dlp, const double tol, const int maxsweeps)
{
	int ierr = 0;

	const int thread_chunk_size = parsePetscCmd_int("-blasted_thread_chunk_size");
	const std::string initialization = parsePetscCmd_string("-initialization", 20);

	const SRMatrixStorage<const double,const int> smat = wrapLocalPetscMat(dlp.lhs, bs);
	printf(" Input problem: Dimension = %d, nnz = %d\n", smat.nbrows*bs, smat.nnzb*bs*bs);
	assert(smat.nnzb == smat.browptr[smat.nbrows]);

	const CRawBSRMatrix<double,int> mat = createRawView(std::move(smat));
	assert(mat.nnzb == mat.browptr[mat.nbrows]);

	const ILUPositions<int> plist = compute_ILU_positions_CSR_CSR(&mat);

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

device_vector<double> getScalingVector(const CRawBSRMatrix<double,int> *const mat)
{
	device_vector<double> scale(mat->nbrows);

#pragma omp parallel for simd default(shared)
	for(int i = 0; i < mat->nbrows; i++)
		scale[i] = 1.0/std::sqrt(mat->vals[mat->diagind[i]]);

	return scale;
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

#if 0
template <int bs>
int run_async_factorization_loop(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                 const double initLerr, const double initUerr,
                                 const double tol, const int maxsweeps, const int thread_chunk_size)
{
	Blk<bs> *const ilu = reinterpret_cast<Blk<bs>*>(&iluvals[0]);
	const Blk<bs> *const mvals = reinterpret_cast<const Blk<bs>*>(mat.vals);

	int isweep = 0;
	double Lerr = 1.0, Uerr = 1.0;

	printf(" %5s %10s %10s\n", "Sweep", "L-norm","U-norm");

#pragma omp parallel default(shared) firstprivate(isweep) reduction(max:Lerr,Uerr)
	while(isweep < maxsweeps && (Lerr > tol || Uerr > tol))
	{
		if(bs == 1)
		{
#pragma omp for default(shared) schedule(dynamic, thread_chunk_size)
			for(int irow = 0; irow < mat.nbrows; irow++)
			{
				async_ilu0_factorize_kernel<double,int,true,true>(&mat, plist, irow,
				                                                  &scale[0], &scale[0], &iluvals[0]);
			}
		}
		else
		{
#pragma omp for default(shared) schedule(dynamic, thread_chunk_size)
			for(int irow = 0; irow < mat.nbrows; irow++)
			{
				async_block_ilu0_factorize<double,int,bs,ColMajor>(&mat, mvals, plist, irow, ilu);
			}
		}

		Lerr = maxnorm_lower<bs>(&mat, iluvals, exactilu)/initLerr;
		Uerr = maxnorm_upper<bs>(&mat, iluvals, exactilu)/initUerr;

		printf(" %5d %10.3g %10.3g\n", isweep, Lerr, Uerr);

		isweep++;
	}
}
#endif
