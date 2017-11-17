/** \file csrmatrix.cpp
 * \brief Tests CSR matrix operations
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
#include "testcsrmatrix.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 2) {
		std::cout << "! Please specify the test. Options:\n";
		std::cout << " apply\n gemv \n";
		std::abort();
	}
	std::string teststr = argv[1];

	int err = 0;

	if(teststr == "apply")
	{
		int rowptr[3] = {0, 1, 2}, colind[2] = {0,1}, diagind[2]={0,1};
		double data[2] = {1,2};
		LinearOperator<double,int>* testmat = nullptr;
		testmat = new BSRMatrix<double,int,1>(2,rowptr,colind,data,diagind,1,1);
		double avec[2] = {1,2}, bvec[2];
		testmat->apply(1.0, avec, bvec);

		assert(bvec[0]==1);
		assert(bvec[1]==4);
	
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
