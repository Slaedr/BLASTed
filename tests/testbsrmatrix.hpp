/** \file testbsrmatrix.hpp
 * \brief Tests for block matrix operations
 * \author Aditya Kashi
 */

#ifndef TESTBSRMATRIX_H
#define TESTBSRMATRIX_H

/// Tests matrix vector product for block matrices and block matrix views
/** \param type "view" or "matrix" depending on what you want to test
 * \param storageorder "rowmajor" or "colmajor" depending on what you want to test
 * \param matfile File name of the mtx file containing the matrix in COO format
 * \param xvec File name of mtx file containing the vector to be multiplied in dense format
 * \param prodvec File name of the mtx file containing the solution vector with which to compare
 */
template <int bs>
int testBSRMatMult(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);

#endif
