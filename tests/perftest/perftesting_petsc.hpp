/** \file
 * \brief Performance tests for BLASTed solvers on a given matrix
 */

#ifndef BLASTED_PERFTESTING_PETSC_H
#define BLASTED_PERFTESTING_PETSC_H

#include <petscmat.h>
#include "perftesting.hpp"

namespace blasted {

/// Runs the solver on a certain matrix and right hand side. Writes a line to the report file
/** Writes out actual time take for reference runs (RunParams::ref), but speedups w.r.t. a reference
 * run for test runs.
 * 
 * \param[in] refdata Timings for a reference run, used to write out speedups with respect to
 * \param[out] td Variables in this are assumed to have zero values.
 */
int run_one_test(const RunParams rp, const TimingData refdata, const Mat A, const Vec b, Vec u,
                 TimingData& td, std::ofstream& report);

}

#endif
