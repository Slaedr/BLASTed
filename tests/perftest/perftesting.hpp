/** \file
 * \brief Performance tests for BLASTed solvers on a given matrix
 */

#ifndef BLASTED_PERFTESTING_H
#define BLASTED_PERFTESTING_H

#include <string>
#include <vector>
#include <fstream>

namespace blasted {

// Field width in the report file
constexpr int field_width = 11;

/// Parameters of a thread scaling test
struct TestParams {
	int refthreads;                 ///< Number of threads to use for reference run
	std::vector<int> threadslist;   ///< Thread-counts to test
	int refnruns;                   ///< Number of times to repeat the reference run
	int nruns;                      ///< Number of times to repeat the test runs
	int nrefbswps;                  ///< Number of build-sweeps to use for the ref run
	int nrefaswps;                  ///< Number of apply-sweeps to use for the ref run
	std::vector<int> nbswpslist;    ///< List of number of build-sweeps to use for test runs
	std::vector<int> naswpslist;    ///< List of number of apply-sweeps to use for test runs
	std::string reportfile;         ///< Path to output file
};

/// Settings characterizing one run
struct RunParams {
	bool ref;              ///< Whether this is a 'reference' run
	int nbswps;            ///< Number of build sweeps to construct the preconditioner
	int naswps;            ///< Number of apply sweeps
	int nrepeats;          ///< Number of times to repeat and average over
	int numthreads;        ///< Number of threads to use
};

struct TimingData {
	int nelem;                   ///< Size of the problem - the number of rows
	int num_threads;             ///< Number of threads used to solve the problem
	double walltime;             ///< Wall-clock time taken by the linear solve
	double cputime;              ///< CPU time taken by the linear solve
	int avg_lin_iters;           ///< Average number of linear iters needed for solve
	int max_lin_iters;           ///< Maximum number of linear iters needed for solve
	bool converged;              ///< Did the nonlinear solver converge?
	double precsetup_walltime;   ///< Custom preconditioner setup wall time
	double precapply_walltime;   ///< Custom preconditioner apply wall time
	double prec_cputime;         ///< Total CPU time taken by custom preconditioner

	/// Convergence history
	std::vector<double> convhis;

	/// Initialize
	TimingData() : walltime{0}, cputime{0}, avg_lin_iters{0}, max_lin_iters{0}, converged{false},
	               precsetup_walltime{0}, precapply_walltime{0}, prec_cputime{0}
	{ }
};

TestParams getTestParams();

/// Runs the solver on a certain matrix and right hand side. Writes a line to the report file
/** Writes out actual time take for reference runs (RunParams::ref), but speedups w.r.t. a reference
 * run for test runs.
 * 
 * \param[in] refdata Timings for a reference run, used to write out speedups with respect to
 * \param[out] td Variables in this are assumed to have zero values.
 */
int run_one_test(const RunParams rp, const TimingData refdata, const Mat A, const Vec b, Vec u,
                 TimingData& td, std::ofstream& report);

/// Writes the line of column headers to the output report file
void writeHeaderToFile(std::ofstream& outf, const int width);

}

#endif
