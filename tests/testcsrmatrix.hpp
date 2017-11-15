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
#include <vector>

using namespace blasted;

template <typename scalar>
class TestCSRMatrix : public BSRMatrix<scalar,int,1>
{
public:
	TestCSRMatrix(const int nbuildsweeps, const int napplysweeps);

	/// Tests the matrix storage
	int testStorage(const std::string compare_file);

protected:
	using BSRMatrix<scalar,int,1>::vals;
	using BSRMatrix<scalar,int,1>::bcolind;
	using BSRMatrix<scalar,int,1>::browptr;
	using BSRMatrix<scalar,int,1>::diagind;
	using BSRMatrix<scalar,int,1>::nbrows;
};

#endif
