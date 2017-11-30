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
		if(argc < 6) {
			std::cout << "! Please provide file names of matrix, x vector and product.\n";
			std::abort();
		}
		int ierr = testCSRMatMult(argv[2], argv[3], argv[4], argv[5]);
		err = ierr || err;
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
