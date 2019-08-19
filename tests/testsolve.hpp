/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#ifndef TESTSOLVE_H
#define TESTSOLVE_H

/// Tests preconditioning operations using a linear solve
/** NOTE: For ILU-type preconditioners, only tests un-scaled variants.
 * \param solvertype The iterative solver to use: "richardson", "bcgs"
 * \param precontype The preconditioner to test: "jacobi", "sgs", "ilu0" or "none"
 * \param factinittype Initial guess method for asynchronous factorizations
 * \param applyinittype Initial guess method for asynchronous preconditioner applications
 * \param mattype The type of matrix to test the preconditioner with: "csr" or "bsr"
 * \param storageorder Matters only for BSR matrices - whether the entries within blocks
 *   are stored "rowmajor" or "colmajor"
 * \param matfile The mtx file containing the matrix
 * \param xfile The mtx file containing the true solution
 * \param bfile The mtx file containing the RHS
 * \param threadchunksize Number of iterations assigned to a thread at a time
 */
template<int bs>
int testSolve(const std::string solvertype, const std::string precontype,
              const std::string factinittype, const std::string applyinittype,
              const std::string mattype, const std::string storageorder, const double testtol,
              const std::string matfile, const std::string xfile, const std::string bfile,
              const double tol, const int maxiter, const int nbuildswps, const int napplyswps,
              const int threadchunksize);

#endif
