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
              const std::string factinittype, const std::string applyinittype,
              const std::string mattype, const std::string storageorder, const double testtol,
              const std::string matfile, const std::string xfile, const std::string bfile,
              const double tol, const int maxiter, const int nbuildswps, const int napplyswps,
              const int threadchunksize)
{
	std::cout << "Inputs: Solver = " <<solvertype 
		<< ", Prec = " << precontype
		<< ", order = " << storageorder << ", test tol = " << testtol 
		<< ", tolerance = " << tol << " maxiter = " << maxiter
		<< ",\n  Num build sweeps = " << nbuildswps << ", num apply sweeps = " << napplyswps << '\n';

	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);

	const device_vector<double> ans = readDenseMatrixMarket<double>(xfile);
	const device_vector<double> b = readDenseMatrixMarket<double>(bfile);

	SRMatrixView<double,int>* mat = nullptr;
	if (bs==1)
		mat = new CSRMatrixView<double,int>(move_to_const<double,int>
		                                    (getSRMatrixFromCOO<double,int,bs>(coom, storageorder)));
	else
		if(storageorder == "rowmajor")
			mat = new BSRMatrixView<double,int,bs,RowMajor>
				(move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(coom, storageorder)));
		else
			mat = new BSRMatrixView<double,int,bs,ColMajor>
				(move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(coom, storageorder)));

	SRMatrixStorage<const double, const int> cmat = move_to_const<double,int>
		(getSRMatrixFromCOO<double,int,bs>(coom, storageorder));

	device_vector<double> x(mat->dim(),0.0);

	// construct preconditioner context

	SRFactory<double,int> fctry;
	SRPreconditioner<double,int>* prec = nullptr;
	// For async preconditioners
	AsyncSolverSettings params;
	params.scale = false;
	params.nbuildsweeps = nbuildswps;
	params.napplysweeps = napplyswps;
	params.thread_chunk_size = threadchunksize;
	params.bs = bs;
	params.prectype = fctry.solverTypeFromString(precontype);
	params.fact_inittype = getFactInitFromString(factinittype);
	params.apply_inittype = getApplyInitFromString(applyinittype);
	if(storageorder == "rowmajor")
		params.blockstorage = RowMajor;
	else
		params.blockstorage = ColMajor;
	params.relax = false;

	// prec = fctry.create_preconditioner(move_to_const<double,int>
	//                                    (getSRMatrixFromCOO<double,int,bs>(coom, storageorder)),
	//                                    params);
	prec = fctry.create_preconditioner(std::move(cmat), params);

	// prec->wrap(mat->getSRStorage().nbrows, &mat->getSRStorage().browptr[0], &mat->getSRStorage().bcolind[0],
	//            &mat->getSRStorage().vals[0], &mat->getSRStorage().diagind[0]);
	prec->compute();

	IterativeSolver* solver = nullptr;
	if(solvertype == "richardson")
		solver = new RichardsonSolver(*mat,*prec);
	else if(solvertype == "bcgs")
		solver = new BiCGSTAB(*mat,*prec);
	else {
		std::cout << " ! Invalid solver option!\n";
		std::abort();
	}

	solver->setParams(tol,maxiter);
	std::cout << "Starting solve " << std::endl;
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

	return 0;
}

template
int testSolve<1>(const std::string solvertype, const std::string precontype,
                 const std::string factinittype, const std::string applyinittype,
                 const std::string mattype, const std::string storageorder, const double testtol,
                 const std::string matfile, const std::string xfile, const std::string bfile,
                 const double tol, const int maxiter, const int nbuildswps, const int napplyswps,
                 const int threadchunksize);

template
int testSolve<4>(const std::string solvertype, const std::string precontype,
                 const std::string factinittype, const std::string applyinittype,
                 const std::string mattype, const std::string storageorder, const double testtol,
                 const std::string matfile, const std::string xfile, const std::string bfile,
                 const double tol, const int maxiter, const int nbuildswps, const int napplyswps,
                 const int threadchunksize);

