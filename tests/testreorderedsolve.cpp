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

class TrivialReorderingScaling : public ReorderingScaling<double,int,1>
{
public:
	void compute(const CRawBSRMatrix<double,int>& mat)
	{
	}
};

TrivialReorderingScaling createTrivialColReordering(const int N)
{
	std::vector<int> v(N);
	for(int i = 0; i < N; i++)
		// v[i] = N-i-1;
	    v[i] = i;

	const int temp = v[1];
	v[1] = v[3]; v[3] = temp;

	TrivialReorderingScaling rs;
	rs.setOrdering(&v[0], &v[0], N);
	return rs;
}

int testSolve(const std::string solvertype,
              const std::string factinittype, const std::string applyinittype,
              const std::string mattype, const std::string storageorder, const double testtol,
              const std::string matfile, const std::string xfile, const std::string bfile,
              const double tol, const int maxiter, const int nbuildswps, const int napplyswps,
              const int threadchunksize)
{
	std::cout << "Inputs: Solver = " <<solvertype 
		<< ", Prec = Async ILU0"
		<< ", order = " << storageorder << ", test tol = " << testtol 
		<< ", tolerance = " << tol << " maxiter = " << maxiter
		<< ",\n  Num build sweeps = " << nbuildswps << ", num apply sweeps = " << napplyswps << '\n';

	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);

	const std::vector<double> ans = readDenseMatrixMarket<double>(xfile);
	const std::vector<double> b = readDenseMatrixMarket<double>(bfile);

	const CSRMatrixView<double,int> mat(move_to_const<double,int>
	                                    (getSRMatrixFromCOO<double,int,1>(coom, storageorder)));

	std::vector<double> x(mat.dim(),0.0);

	// reference solve - no scaling
	SRPreconditioner<double,int>* prec = nullptr;
	prec = new AsyncILU0_SRPreconditioner<double,int>(move_to_const<double,int>
	                                                  (getSRMatrixFromCOO<double,int,1>(coom, storageorder)),
	                                                  nbuildswps, napplyswps, false,
	                                                  threadchunksize,
	                                                  getFactInitFromString(factinittype),
	                                                  getApplyInitFromString(applyinittype),
	                                                  false, false);
	// prec->wrap(mat.getSRStorage().nbrows, &mat.getSRStorage().browptr[0], &mat.getSRStorage().bcolind[0],
	//            &mat.getSRStorage().vals[0], &mat.getSRStorage().diagind[0]);
	prec->compute();

	IterativeSolver* solver = nullptr;
	if(solvertype == "richardson")
		solver = new RichardsonSolver(mat,*prec);
	else if(solvertype == "bcgs")
		solver = new BiCGSTAB(mat,*prec);
	else {
		std::cout << " ! Invalid solver option!\n";
		std::abort();
	}

	solver->setParams(tol,maxiter);
	const int refiters = solver->solve(b.data(), x.data());
	std::cout << " Ref num iters = " << refiters << std::endl;

	double refl2norm = 0;
	for(int i = 0; i < mat.dim(); i++) {
		refl2norm += (x[i]-ans[i])*(x[i]-ans[i]);
	}
	refl2norm = std::sqrt(refl2norm);
	std::cout << " Ref L2 norm of error = " << refl2norm << '\n';
	assert(refl2norm < testtol);
	delete solver;
	delete prec;

	// solve with reordered preconditioning
	std::cout << " Prec = ReorderedAsyncILU0" << std::endl;

	x.assign(mat.dim(), 0.0);

	TrivialReorderingScaling rs = createTrivialColReordering(mat.dim());
	prec = new ReorderedAsyncILU0_SRPreconditioner<double,int>
		(move_to_const<double,int>(getSRMatrixFromCOO<double,int,1>(coom, storageorder)),
		 &rs, nbuildswps, napplyswps, threadchunksize,
		 getFactInitFromString(factinittype), getApplyInitFromString(applyinittype), false, false);

	// prec->wrap(mat.getSRStorage().nbrows, &mat.getSRStorage().browptr[0], &mat.getSRStorage().bcolind[0],
	//            &mat.getSRStorage().vals[0], &mat.getSRStorage().diagind[0]);
	prec->compute();

	if(solvertype == "richardson")
		solver = new RichardsonSolver(mat,*prec);
	else if(solvertype == "bcgs")
		solver = new BiCGSTAB(mat,*prec);
	else {
		std::cout << " ! Invalid solver option!\n";
		std::abort();
	}

	solver->setParams(tol,maxiter);
	const int iters = solver->solve(b.data(), x.data());
	std::cout << " Num iters = " << iters << std::endl;

	double l2norm = 0;
	for(int i = 0; i < mat.dim(); i++) {
		l2norm += (x[i]-ans[i])*(x[i]-ans[i]);
	}
	l2norm = std::sqrt(l2norm);
	std::cout << " L2 norm of error = " << l2norm << std::endl;
	assert(l2norm < testtol);

	delete solver;
	delete prec;

	return 0;
}

int main(const int argc, const char *const argv[])
{
	if(argc < 14) {
		std::cout << "! Please specify the solver (richardson, bcgs),\n";
		std::cout << " the preconditioner (options: jacobi, sgs, ilu0), \n";
		std::cout << " the factor initialization type (options: init_zero, init_sgs, init_original)\n";
		std::cout << " the apply initialization type (options: init_zero, init_jacobi)\n";
		std::cout << " the matrix type to use (options: csr, bsr),\n";
		std::cout << "whether the entries within blocks should be rowmajor or colmajor\n";
		std::cout << "(this option does not matter for CSR, but it's needed anyway),\n";
		std::cout << "the three file names of (in order) the matrix,\n"
		          << "the true solution vector x and the RHS vector b,\n"
		          << "the rel residual tolerance to which to solve the linear system,\n"
		          << "the testing tolerance for judging correctness,\n"
		          << "the max number of iterations,\n"
		          << "and the thread chunk size.\n";
		std::abort();
	}
	
	const int maxiter = std::stoi(argv[12]);
	const double reltol = std::stod(argv[10]);
	const double testtol = std::stod(argv[11]);
	const int threadchunksize = std::stoi(argv[13]);
	const std::string solvertype = argv[1];
	const std::string precontype = argv[2];
	const std::string mattype = argv[5];
	const std::string storageorder = argv[6];
	const std::string factinittype = argv[3];
	const std::string applyinittype = argv[4];
	const std::string matfile = argv[7];
	const std::string xfile = argv[8];
	const std::string bfile = argv[9];

	int err = testSolve(solvertype, factinittype, applyinittype, mattype, storageorder, 
	                    testtol, matfile, xfile, bfile, reltol, maxiter, 1, 1, threadchunksize);

	return err;
}
