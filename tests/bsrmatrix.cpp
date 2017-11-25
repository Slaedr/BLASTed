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
		int rowptr[4] = {0, 2, 3, 4}, colind[4] = {0,2,1,2}, diagind[3]={0,2,3};
		double rdata[36] = { 1,2,5,3,0,1,2,6,0,
			3,1,0,2,6,3,1,1,1,
			0.5,2,-6,1,6,2,2,1,9,
			10,1,1.5,0,-8,2,1,3,5
		};
		double cdata[36] = {1,3,2,2,0,6,5,1,0,
			3,2,1,1,6,1,0,3,1,
			0.5,1,2,2,6,1,-6,2,9,
			10,0,1,1,-8,3,1.5,2,5
		};
		double x[9] = {1,2,1,1,3,1,3,2,-1};
		double ans[9] = {21,19,18,0.5,21,14,30.5,-18,4};
		double y[9];

		AbstractLinearOperator<double,int>* testmat = nullptr;
		if(typestr == "view") {
			testmat = new BSRMatrixView<double,int,3,RowMajor>(3,rowptr,colind,rdata,diagind,1,1);
		}
		else
			testmat = new BSRMatrix<double,int,3>(3,rowptr,colind,rdata,diagind,1,1);
		
		testmat->apply(1.0, x, y);

		for(int i = 0; i < 9; i++) {
			assert(y[i] == ans[i]);
		}

		delete testmat;
	
		// Test col-major within the blocks as well
		if(typestr == "view") 
		{
			for(int i = 0; i < 9; i++) {
				y[i] = 0;
			}

			testmat = new BSRMatrixView<double,int,3,ColMajor>(3,rowptr,colind,cdata,diagind,1,1);
			testmat->apply(1.0, x, y);

			for(int i = 0; i < 9; i++) {
				assert(y[i] == ans[i]);
			}
			
			delete testmat;
		}
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
