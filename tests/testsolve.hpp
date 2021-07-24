/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#ifndef BLASTED_TESTSOLVE_H
#define BLASTED_TESTSOLVE_H

#include <string>

namespace blasted_testsolve {

struct Params {
  std::string solvertype;    ///< The iterative solver to use: "richardson", "bcgs"
  std::string precontype;    ///< The preconditioner to test: "jacobi", "sgs", "ilu0", "none"
  std::string factinittype;  ///< Initial guess method for asynchronous factorizations
  std::string applyinittype; ///< Initial guess method for asynchronous preconditioner applications
  std::string mattype;       ///< The type of matrix to test the preconditioner with: "csr" or "bsr"
  int blocksize;             ///< Block size in the case of BSR matris format
  /** Matters only for BSR matrices - whether the entries within blocks
   * are stored "rowmajor" or "colmajor".
   */
  std::string storageorder;
  double testtol;            ///< Tolerance for solution vector
  double tol;                ///< Residual tolerance
  int maxiter;               ///< Max no. of solver iterations
  int solver_restart;        ///< For some solvers, number of subspace vectors before restart
  int threadchunksize;
  int nbuildsweeps;
  int napplysweeps;
  std::string mat_file;
  std::string x_file;
  std::string b_file;
};

/// Reads test solve parameters from command line
Params read_from_cmd(int argc, const char *const argv[]);

/// Tests preconditioning operations using a linear solve
/** NOTE: For ILU-type preconditioners, only tests un-scaled variants.
 * \param params  List of parameters to use for solver and preconditioner.
 */
template<int bs>
int test_solve(const Params params);

}

#endif
