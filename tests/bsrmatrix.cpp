/** \file bsrmatrix.cpp
 * \brief Tests block matrix operations
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

#include <iostream>
#include "testbsrmatrix.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 3) {
		std::cout << "! Please specify the test (options:  apply , gemv) \n";
		std::cout << " and whether a matrix or a matrix view is to be tested.\n";
		std::abort();
	}

	// The test to execute
	std::string teststr = argv[1];
	// Whether to do it on a matrix or a matrix view
	std::string typestr = argv[2];

	int err = 0;

	if(teststr == "apply")
	{
		int rowptr[3] = {0, 1, 2}, colind[2] = {0,1}, diagind[2]={0,1};
		double data[8] = {1,0,0,1,2,0,0,2};
		AbstractLinearOperator<double,int>* testmat = nullptr;
		if(typestr == "view")
			testmat = new BSRMatrixView<double,int,2>(2,rowptr,colind,data,diagind,1,1);
		else
			testmat = new BSRMatrix<double,int,2>(2,rowptr,colind,data,diagind,1,1);
		double avec[4] = {1,2,3,4}, bvec[4];
		testmat->apply(1.0, avec, bvec);

		assert(bvec[0]==1);
		assert(bvec[1]==2);
		assert(bvec[2]==6);
		assert(bvec[3]==8);
	
		delete testmat;
	}
	else if(teststr == "gemv")
	{
		std::cout << "Not implemented yet..\n";
	}
	else {
		std::cout << "! The requested test is not available.\n";
		std::abort();
	}

	return err;
}
