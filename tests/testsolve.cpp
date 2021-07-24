/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#undef NDEBUG

#include <iostream>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

#include <blockmatrices.hpp>
#include <coomatrix.hpp>
#include <solverfactory.hpp>
#include <solverops_jacobi.hpp>
#include <solverops_sgs.hpp>
#include <solverops_ilu0.hpp>

#include "testsolve.hpp"
#include "solvers.hpp"

using namespace blasted;

namespace blasted_testsolve {

template<int bs>
int test_solve(const Params params)
{
	std::cout << "Inputs: Solver = " << params.solvertype
		<< ", Prec = " <<  params.precontype
		<< ", order = " <<  params.storageorder << ", test tol = " <<  params.testtol
		<< ", tolerance = " <<  params.tol << " maxiter = " <<  params.maxiter
		<< ",\n  Num build sweeps = " <<  params.nbuildsweeps
			  << ", num apply sweeps = " << params.napplysweeps << '\n';

	COOMatrix<double,int> coom;
	coom.readMatrixMarket(params.mat_file);

	const device_vector<double> ans = readDenseMatrixMarket<double>(params.x_file);
	const device_vector<double> b = readDenseMatrixMarket<double>(params.b_file);

	SRMatrixView<double,int>* mat = nullptr;
	if (bs==1)
		mat = new CSRMatrixView<double,int>(
			move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(
										coom, params.storageorder)));
	else
		if(params.storageorder == "rowmajor")
			mat = new BSRMatrixView<double,int,bs,RowMajor>(
			    move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(
											coom, params.storageorder)));
		else
			mat = new BSRMatrixView<double,int,bs,ColMajor>(
				move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(
											coom, params.storageorder)));

	SRMatrixStorage<const double, const int> cmat = move_to_const<double,int>
		(getSRMatrixFromCOO<double,int,bs>(coom, params.storageorder));

	device_vector<double> x(mat->dim(),0.0);

	// construct preconditioner context

	SRFactory<double,int> fctry;
	// For async preconditioners
	AsyncSolverSettings aparams;
	aparams.scale = false;
	aparams.nbuildsweeps = params.nbuildsweeps;
	aparams.napplysweeps = params.napplysweeps;
	aparams.thread_chunk_size = params.threadchunksize;
	aparams.bs = bs;
	aparams.prectype = fctry.solverTypeFromString(params.precontype);
	aparams.fact_inittype = getFactInitFromString(params.factinittype);
	aparams.apply_inittype = getApplyInitFromString(params.applyinittype);
	if(params.storageorder == "rowmajor")
		aparams.blockstorage = RowMajor;
	else
		aparams.blockstorage = ColMajor;
	aparams.relax = false;

	const auto prec = fctry.create_preconditioner(std::move(cmat), aparams);

	prec->compute();

	IterativeSolver* solver = nullptr;
	if(params.solvertype == "richardson")
		solver = new RichardsonSolver(*mat,*prec);
	else if(params.solvertype == "bcgs")
		solver = new BiCGSTAB(*mat,*prec);
	else if(params.solvertype == "gcr")
		solver = new GCR(*mat,*prec, params.solver_restart);
	else {
		std::cout << " ! Invalid solver option!\n";
		std::abort();
	}

	solver->setParams(params.tol,params.maxiter);
	std::cout << "Starting solve " << std::endl;
	const auto info = solver->solve(b.data(), x.data());
	std::cout << " Num iters = " << info.iters << std::endl;

	double l2norm = 0;
	for(int i = 0; i < mat->dim(); i++) {
		l2norm += (x[i]-ans[i])*(x[i]-ans[i]);
	}
	l2norm = std::sqrt(l2norm);
	std::cout << " L2 norm of error = " << l2norm << '\n';
	assert(l2norm < params.testtol);

	delete solver;
	delete prec;
	delete mat;

	return 0;
}

template int test_solve<1>(const Params params);
template int test_solve<4>(const Params params);
#ifdef BUILD_BLOCK_SIZE
template int test_solve<BUILD_BLOCK_SIZE>(const Params params);
#endif

namespace po = boost::program_options;

Params read_from_cmd(const int argc, const char *const argv[])
{
	Params p;
	po::options_description desc("Options available for test solve with inbuilt solvers");
    desc.add_options()
        ("help", "Print help message")
		("solver_type", po::value<std::string>(&p.solvertype)->default_value("bcgs"),
		     "Solver to use")
		("preconditioner_type", po::value<std::string>(&p.precontype)->default_value("jacobi"),
		     "Preconditioner to use")
		("fact_init_type",
		     po::value<std::string>(&p.factinittype)->default_value("init_original"),
		     "Type of initial values for iterative preconditioner factorization")
		("apply_init_type",
		     po::value<std::string>(&p.applyinittype)->default_value("init_zero"),
		     "Type of initial values for iterative preconditioner application")
		("mat_type", po::value<std::string>(&p.mattype)->default_value("csr"),
		     "Type (format) of matrix for the solver/preconditioner to be given")
		("block_size", po::value<int>(&p.blocksize)->default_value(4),
		     "Block size to use in the case of BSR format")
		("storage_order", po::value<std::string>(&p.storageorder)->default_value("colmajor"),
		     "Block layout to use in case of bsr matrix type")
		("test_tol", po::value<double>(&p.testtol)->default_value(1e-4), "Tolerance on solution")
		("solver_tol", po::value<double>(&p.tol)->default_value(1e-6),
		     "Relative residual tolerance for solver convergence")
		("max_iter", po::value<int>(&p.maxiter)->default_value(1000), "Maximum solver iteratons")
		("solver_restart", po::value<int>(&p.solver_restart)->default_value(30),
		 "For certain solvers, number of subspace vectors to build before restarting")
		("build_sweeps", po::value<int>(&p.nbuildsweeps)->default_value(1),
		     "Number of sweeps for iterative factorization of preconditioner")
		("apply_sweeps", po::value<int>(&p.napplysweeps)->default_value(1),
		     "Number of sweeps for iterative application of preconditioner within the solver")
		("thread_chunk_size", po::value<int>(&p.threadchunksize)->default_value(256),
		     "Number of work-items in each chunk of work given to a thread")
		("mat_file", po::value<std::string>(&p.mat_file),
		     "Path to matrix-market file for system matrix")
		("b_file", po::value<std::string>(&p.b_file),
		     "Path to matrix-market file for right-hand-side vector")
		("x_file", po::value<std::string>(&p.x_file),
		     "Path to matrix-market file for reference solution vector")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
		p.maxiter = -1;
    }

	return p;
}

}
