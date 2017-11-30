/** \file runbsrmatrixtest.cpp
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

#undef NDEBUG
#include <iostream>
#include "testbsrmatrix.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 5) {
		std::cout << "! Please specify the test (options:  apply , gemv), \n";
		std::cout << " whether a matrix or a matrix view is to be tested,\n";
		std::cout << "whether the entries within blocks should be rowmajor or colmajor,\n"
			<< "and the block size.\n";
		std::abort();
	}

	// The test to execute
	const std::string teststr = argv[1];
	// Whether to do it on a matrix or a matrix view
	const std::string typestr = argv[2];
	// Whether the blocks are rowmajor or colmajor
	const std::string bstorstr = argv[3];

	int bs;
	try {
		bs = std::stoi(argv[4]);
	}
	catch(const std::invalid_argument& e) {
		std::cout << "! Invalid block size!!\n";
		std::abort();
	}
	catch(const std::out_of_range& e) {
		std::cout << "! Block size is too large for the type of index!!\n";
		std::abort();
	}

	int err = 0;

	if(teststr == "apply")
	{
		if(argc < 8) {
			std::cout<< "! Please give filenames for a block matrix with block size 7,";
			std::cout << "the vector and the solution vector.\n";
			std::abort();
		}
		int ierr = 0;
		if(bs == 7)
			ierr = testBSRMatMult<7>(typestr, bstorstr, argv[5], argv[6], argv[7]);
		else if(bs == 3)
			ierr = testBSRMatMult<3>(typestr, bstorstr, argv[5], argv[6], argv[7]);
		else {
			std::cout << "Block size not supported!\n";
			std::abort();
		}
		err = err || ierr;

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
