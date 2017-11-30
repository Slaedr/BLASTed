/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#ifndef TESTSOLVE_H
#define TESTSOLVE_H

/// Tests preconditioning operations using a Richardson solve
/** \param precontype The preconditioner to test: "jacobi", "sgs" or "ilu0"
 * \param mattype The type of matrix to test the preconditioner with: "csr" or "bsr"
 * \param storageorder Matters only for BSR matrices - whether the entries within blocks
 *   are stored "rowmajor" or "colmajor"
 * \param matfile The mtx file containing the matrix
 * \param xfile The mtx file containing the true solution
 * \param bfile The mtx file containing the RHS
 */
template<int bs>
int testSolveRichardson(const std::string precontype,
		const std::string mattype, const std::string storageorder, const double testtol,
		const std::string matfile, const std::string xfile, const std::string bfile,
		const double tol, const int maxiter, const int nbuildswps, const int napplyswps);

#endif
