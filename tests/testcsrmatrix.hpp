/** \file testcsrmatrix.hpp
 * \brief Tests for CSR matrix operations
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TESTCSRMATRIX_H
#define TESTCSRMATRIX_H

#include "../src/blockmatrices.hpp"

using namespace blasted;

template <typename scalar>
class TestCSRMatrix : public BSRMatrix<scalar,int,1>
{
public:
	TestCSRMatrix(const int nbuildsweeps, const int napplysweeps);

	/// Tests the matrix storage
	int testStorage(const std::string compare_file);

protected:
	using BSRMatrix<scalar,int,1>::mat;
};

/// Tests matrix vector product for CSR matrices and views
/** \param type "view" or "matrix" depending on what you want to test
 * \param matfile File name of the mtx file containing the matrix in COO format
 * \param xvec File name of mtx file containing the vector to be multiplied in dense format
 * \param prodvec File name of the mtx file containing the solution vector with which to compare
 */
int testCSRMatMult(const std::string type, 
		const std::string matfile, const std::string xvec, const std::string prodvec);

#endif
