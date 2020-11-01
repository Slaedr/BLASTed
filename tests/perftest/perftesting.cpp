/** \file
 * \brief Implementation of a thread-scaling test for BLASTed solvers on a given matrix
 */

#include <iomanip>
#include <stdexcept>
#include <petscksp.h>

#include "iterative_preconditioner.hpp"
#include "perftesting.hpp"

#define MAX_THREADS_LIST_SIZE 20
#define PATH_STR_LEN 500

namespace blasted {

/// Set -blasted_async_sweeps in the default Petsc options database and throw if not successful
template <typename PrecType>
static void set_blasted_sweeps(const int nbswp, const int naswp, PrecType& prec)
{
	IterPrecParams aparams = prec.getParams();
	aparams.nbuildsweeps = nbswp;
	aparams.napplysweeps = naswp;
	prec.setAsyncParams(aparams);
}

void writeHeaderToFile(std::ofstream& outf, const int width)
{
	outf << '#' << std::setw(width) << "threads"
	     << std::setw(width) << "b&a-sweeps"
	     << std::setw(width+5) << "factor-speedup" << std::setw(width+5) << "apply-speedup"
	     << std::setw(width+5) << "total-speedup" << std::setw(width+5) << "total-deviate"
	     << std::setw(width) << "cpu-time" << std::setw(width+6) << "max-lin-iters"
	     << std::setw(width+4) << "avg-lin-iters"
	     << std::setw(width/2+1) << "conv?"
	     << "\n#---\n";
}

static void writeTimingToFile(std::ofstream& outf, const int w, const bool comment,
                              const TimingData& tdata, const RunParams rp,
                              const double factorspeedup, const double applyspeedup,
                              const double precspeedup, const double precdeviate,
                              const double preccputime)
{
	outf << (comment ? '#' : ' ') << std::setw(w) << rp.numthreads
	     << std::setw(w/2) << rp.nbswps << std::setw(w/2) << rp.naswps
	     << std::setw(w+5) << factorspeedup << std::setw(w+5) << applyspeedup
	     << std::setw(w+5) << precspeedup << std::setw(w+5) << precdeviate
	     << std::setw(w) << preccputime
	     << std::setw(w+6) << tdata.max_lin_iters << std::setw(w+4) << tdata.avg_lin_iters
	     << std::setw(w/2+1) << (tdata.converged ? 1 : 0)
	     << (comment ? "\n#---\n" : "\n") << std::flush;
}

static double std_deviation(const double *const vals, const double avg, const int N) {
	double deviate = 0;
	for(int j = 0; j < N; j++)
		deviate += (vals[j]-avg)*(vals[j]-avg);
	deviate = std::sqrt(deviate/(double)N);
	return deviate;
}

template <typename scalar, typename index = int>
int run_one_test(const RunParams rp, const TimingData refdata, const MatrixView<scalar,index>& AA,
                 const Vector<scalar>& bb, Vector<scalar>& uu,
                 TimingData& tdata, std::ofstream& outf)
{
	std::vector<double> prectimes(rp.nrepeats);
	int ierr = 0;
	MPI_Comm comm = MPI_COMM_WORLD;

	int rank = 1;
	MPI_Comm_rank(comm,&rank);

	SRPreconditioner<scalar,index> prec;

	set_blasted_sweeps(rp.nbswps, rp.naswps, prec);

	Vec u, b; Mat A;

	int irun;
	for(irun = 0; irun < rp.nrepeats; irun++)
	{
		KSP ksp;
		ierr = KSPCreate(comm, &ksp);
		if(rp.ref)
			KSPSetOptionsPrefix(ksp, "ref_");
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

		// Blasted_data_list bctx = newBlastedDataList();
		// ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

		ierr = VecSet(u, 0.0); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		PetscInt kspiters;
		ierr = KSPGetIterationNumber(ksp, &kspiters);
		tdata.avg_lin_iters += kspiters;
		if(tdata.max_lin_iters < kspiters)
			tdata.max_lin_iters = kspiters;

		KSPConvergedReason reason;
		ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
		if(reason <= 0) {
			tdata.converged = false;
			if(rank == 0)
				printf(" Solve failed with error %d", reason);
			break;
		}
		else {
			tdata.converged = true;
		}

		//computeTotalTimes(&bctx);
		//tdata.precsetup_walltime += bctx.factorwalltime;
		//tdata.precapply_walltime += bctx.applywalltime;
		//tdata.prec_cputime += bctx.factorcputime + bctx.applycputime;
		//prectimes[irun] = bctx.factorwalltime + bctx.applywalltime;

		KSPDestroy(&ksp);
		//destroyBlastedDataList(&bctx);
	}

	tdata.precsetup_walltime /= irun;
	tdata.precapply_walltime /= irun;
	tdata.prec_cputime /= irun;
	tdata.avg_lin_iters = static_cast<int>((float)tdata.avg_lin_iters/(float)irun);

	const double prec_deviation = std_deviation(&prectimes[0],
	                                            tdata.precsetup_walltime+tdata.precapply_walltime,
	                                            irun);

	if(rp.ref)
		writeTimingToFile(outf, field_width, rp.ref, tdata, rp,
		                  tdata.precsetup_walltime,
		                  tdata.precapply_walltime,
		                  (tdata.precsetup_walltime+tdata.precapply_walltime),
		                  prec_deviation, tdata.prec_cputime);
	else
		writeTimingToFile(outf, field_width, rp.ref, tdata, rp,
		                  refdata.precsetup_walltime/tdata.precsetup_walltime,
		                  refdata.precapply_walltime/tdata.precapply_walltime,
		                  (refdata.precsetup_walltime+refdata.precapply_walltime)
		                  /(tdata.precsetup_walltime+tdata.precapply_walltime),
		                  prec_deviation, tdata.prec_cputime);

	return ierr;
}

TestParams getTestParams()
{
	TestParams tp;
	int ierr = 0;

	PetscBool set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_ref_threads", &tp.refthreads, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of ref threads not set!");
	}

	set = PETSC_FALSE;
	int threadslist[MAX_THREADS_LIST_SIZE];
	int len = MAX_THREADS_LIST_SIZE;
	ierr = PetscOptionsGetIntArray(NULL,NULL,"-perftest_test_threadslist",threadslist, &len, &set);
	if(ierr || !set) {
		throw std::runtime_error("Need list of thread counts to test!");
	}
	tp.threadslist.resize(len);
	for(int i = 0; i < len; i++)
		tp.threadslist[i] = threadslist[i];

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_ref_runs", &tp.refnruns, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of ref runs not set!");
	}

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_test_runs", &tp.nruns, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of testing runs not set!");
	}

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_ref_build_sweeps", &tp.nrefbswps, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of ref build sweeps not set!");
	}

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_ref_apply_sweeps", &tp.nrefaswps, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of ref apply sweeps not set!");
	}

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_test_build_sweeps", &tp.nbswps, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of testing build sweeps not set!");
	}

	set = PETSC_FALSE;
	ierr = PetscOptionsGetInt(NULL,NULL,"-perftest_test_apply_sweeps", &tp.naswps, &set);
	if(ierr || !set) {
		throw std::runtime_error("Number of testing apply sweeps not set!");
	}

	set = PETSC_FALSE;
	char filename[PATH_STR_LEN];
	ierr = PetscOptionsGetString(NULL,NULL,"-perftest_report_file", filename, PATH_STR_LEN, &set);
	if(ierr || !set) {
		throw std::runtime_error("Path to report output not set!");
	}
	tp.reportfile = filename;

	return tp;
}

}
