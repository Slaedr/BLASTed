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
#include <stdexcept>
#include "testsolve.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 11) {
		std::cout << "! Please specify the solver (richardson, bcgs),\n";
		std::cout << " the preconditioner (options: jacobi, sgs, ilu0), \n";
		std::cout << " the matrix type to use (options: csr, bsr),\n";
		std::cout << "whether the entries within blocks should be rowmajor or colmajor\n";
		std::cout << "(this option does not matter for CSR, but it's needed anyway),\n";
		std::cout << "the three file names of (in order) the matrix,\n"
			<< "the true solution vector x and the RHS vector b,\n"
			<< "the rel residual tolerance to which to solve the linear system,\n"
			<< "the testing tolerance for judging correctness,\n"
			<< "and the max number of iterations.\n";
		std::abort();
	}
	
	int maxiter;
	try {
		maxiter = std::stoi(argv[10]);
	}
	catch(const std::invalid_argument& e) {
		std::cout << "! getSizeFromMatrixMarket: Invalid size!!\n";
		std::abort();
	}
	catch(const std::out_of_range& e) {
		std::cout << "! getSizeFromMatrixMarket: Size is too large for the type of index!!\n";
		std::abort();
	}

	double reltol, testtol;
	try {
		reltol = std::stod(argv[8]);
		testtol = std::stod(argv[9]);
	}
	catch(const std::invalid_argument& e) {
		std::cout << "! Invalid rel or test tol!!\n";
		std::abort();
	}
	catch(const std::out_of_range& e) {
		std::cout << "! Rel or test tol is out of range!!\n";
		std::abort();
	}

	int err = testSolve<4>(argv[1], argv[2], argv[3], argv[4], 
			testtol, argv[5], argv[6], argv[7], reltol, maxiter, 1, 1);

	return err;
}
