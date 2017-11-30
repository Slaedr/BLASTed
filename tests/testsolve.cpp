/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#undef NDEBUG

#include <iostream>
#include "testsolve.hpp"
#include "solvers.hpp"
#include "../src/blockmatrices.hpp"
#include "../src/coomatrix.hpp"

using namespace blasted;

template<int bs>
int testSolveRichardson(const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps)
{
	std::cout << "Inputs: Prec = " << precontype << ", mat type = " << mattype
		<< ", order = " << storageorder << ", test tol = " << testtol 
		<< ", tolerance = " << tol << " maxiter = " << maxiter
		<< ",\n  Num build sweeps = " << nbuildswps << ", num apply sweeps = " << napplyswps << '\n';

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
	for(int i = 0; i < rm.nbrows*bs; i++)
		x[i] = 0;

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

	Preconditioner* prec = nullptr;
	if(precontype == "jacobi")
		prec = new Jacobi(mat);
	else if(precontype == "sgs")
		prec = new SGS(mat);
	else if(precontype == "ilu0")
		prec = new ILU0(mat);
	else {
		std::cout << " ! Invalid preconditioner option!\n";
		std::abort();
	}

	IterativeSolver* solver = new RichardsonSolver(mat,prec);

	solver->setupPreconditioner();
	solver->setParams(tol,maxiter);
	int iters = solver->solve(b, x);
	std::cout << " Num iters = " << iters << std::endl;

	double l2norm = 0;
	for(int i = 0; i < rm.nbrows*bs; i++) {
		l2norm += (x[i]-ans[i])*(x[i]-ans[i]);
	}
	l2norm = std::sqrt(l2norm);
	std::cout << " L2 norm of error = " << l2norm << '\n';
	assert(l2norm < testtol);

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
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

template
int testSolveRichardson<7>(const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

