/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#undef NDEBUG

#include <iostream>

#include <blockmatrices.hpp>
#include <coomatrix.hpp>
#include <solverfactory.hpp>
#include <solverops_jacobi.hpp>
#include <solverops_sgs.hpp>
#include <solverops_ilu0.hpp>

#include "testsolve.hpp"
#include "solvers.hpp"

using namespace blasted;

template<int bs>
int testSolve(const std::string solvertype, const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps)
{
	std::cout << "Inputs: Solver = " <<solvertype 
		<< ", Prec = " << precontype
		<< ", order = " << storageorder << ", test tol = " << testtol 
		<< ", tolerance = " << tol << " maxiter = " << maxiter
		<< ",\n  Num build sweeps = " << nbuildswps << ", num apply sweeps = " << napplyswps << '\n';

	RawBSRMatrix<double,int> rm;
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);
	//if(mattype == "csr")
	if (bs == 1)
		coom.convertToCSR(&rm);
	else
		if(storageorder == "rowmajor")
			coom.convertToBSR<bs,RowMajor>(&rm);
		else
			coom.convertToBSR<bs,ColMajor>(&rm);

	const std::vector<double> ans = readDenseMatrixMarket<double>(xfile);
	const std::vector<double> b = readDenseMatrixMarket<double>(bfile);
	std::vector<double> x(rm.nbrows*bs,0.0);

	MatrixView<double,int>* mat = nullptr;
	//if(mattype == "csr")
	if (bs==1)
		mat = new CSRMatrixView<double,int>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);
	else
		if(storageorder == "rowmajor")
			mat = new BSRMatrixView<double,int,bs,RowMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);
		else
			mat = new BSRMatrixView<double,int,bs,ColMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,nbuildswps,napplyswps);

	// construct preconditioner context
	
	SRPreconditioner<double,int>* prec = nullptr;
	std::map<std::string,int> iparamlist;
	std::map<std::string,double> fparamlist;
	// For async preconditioners
	iparamlist[blasted::nbuildsweeps] = nbuildswps; iparamlist[blasted::napplysweeps] = napplyswps;
	// for no preconditioner
	iparamlist[blasted::ndimstr] = rm.nbrows*bs;
	prec = create_sr_preconditioner<double,int>(precontype, bs, storageorder, false,
	                                            iparamlist, fparamlist);
	prec->wrap(rm.nbrows, rm.browptr, rm.bcolind, rm.vals, rm.diagind);

	IterativeSolver* solver = nullptr;
	if(solvertype == "richardson")
		solver = new RichardsonSolver(*mat,*prec);
	else if(solvertype == "bcgs")
		solver = new BiCGSTAB(*mat,*prec);
	else {
		std::cout << " ! Invalid solver option!\n";
		std::abort();
	}

	//solver->setupPreconditioner();
	solver->setParams(tol,maxiter);
	int iters = solver->solve(b.data(), x.data());
	std::cout << " Num iters = " << iters << std::endl;

	double l2norm = 0;
	for(int i = 0; i < mat->dim(); i++) {
		l2norm += (x[i]-ans[i])*(x[i]-ans[i]);
	}
	l2norm = std::sqrt(l2norm);
	std::cout << " L2 norm of error = " << l2norm << '\n';
	assert(l2norm < testtol);

	delete solver;
	delete prec;
	delete mat;
	//alignedDestroyRawBSRMatrix(rm);
	destroyRawBSRMatrix(rm);

	return 0;
}

template
int testSolve<1>(const std::string solvertype, const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

template
int testSolve<4>(const std::string solvertype, const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

/*template
int testSolve<7>(const std::string solvertype, const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);
*/

