/** \file solvetest.cpp
 * \brief Driver for testing preconditioning operations
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
#include "testsolve.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 7) {
		std::cout << "! Please specify the test (options: jacobi, sgs, ilu0), \n";
		std::cout << " the matrix type to use (options: csr, bsr),\n";
		std::cout << "whether the entries within blocks should be rowmajor or colmajor\n";
		std::cout << "(this option does not matter for CSR, but it's needed anyway),\n";
		std::cout << "and three file names of (in order) the matrix, the true solution vector x\
			 and the RHS vector b.\n";
		std::abort();
	}

	// The test to execute
	const std::string teststr = argv[1];
	// Whether to do it on a CSR or BSR matrix
	const std::string typestr = argv[2];
	// Whether the blocks are rowmajor or colmajor
	const std::string orderstr = argv[3];

	int err = testSolveRichardson<7>(teststr, typestr, orderstr, 1e-4, argv[4], argv[5], argv[6],
			1e-6, 500, 1, 1);

	return err;
}
