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
	if(argc < 14) {
		std::cout << "! Please specify the solver (richardson, bcgs),\n";
		std::cout << " the preconditioner (options: jacobi, sgs, ilu0), \n";
		std::cout << " the factor initialization type (options: init_zero, init_sgs, init_original)\n";
		std::cout << " the apply initialization type (options: init_zero, init_jacobi)\n";
		std::cout << " the matrix type to use (options: csr, bsr),\n";
		std::cout << "whether the entries within blocks should be rowmajor or colmajor\n";
		std::cout << "(this option does not matter for CSR, but it's needed anyway),\n";
		std::cout << "the three file names of (in order) the matrix,\n"
		          << "the true solution vector x and the RHS vector b,\n"
		          << "the rel residual tolerance to which to solve the linear system,\n"
		          << "the testing tolerance for judging correctness,\n"
		          << "the max number of iterations,\n"
		          << "and the thread chunk size.\n";
		std::abort();
	}
	
	const int maxiter = std::stoi(argv[12]);
	const double reltol = std::stod(argv[10]);
	const double testtol = std::stod(argv[11]);
	const int threadchunksize = std::stoi(argv[13]);
	const std::string solvertype = argv[1];
	const std::string precontype = argv[2];
	const std::string mattype = argv[5];
	const std::string storageorder = argv[6];
	const std::string factinittype = argv[3];
	const std::string applyinittype = argv[4];
	const std::string matfile = argv[7];
	const std::string xfile = argv[8];
	const std::string bfile = argv[9];

	int err = 0;
	if(mattype == "bsr")
		err = testSolve<4>(solvertype, precontype, factinittype, applyinittype, mattype, storageorder, 
		                   testtol, matfile, xfile, bfile, reltol, maxiter, 1, 1, threadchunksize);
	else
		err = testSolve<1>(solvertype, precontype, factinittype, applyinittype, mattype, storageorder, 
		                   testtol, matfile, xfile, bfile, reltol, maxiter, 1, 1, threadchunksize);
	/*err = testSolve<1>(argv[1], argv[2], argv[3], argv[4], 
	  testtol, argv[5], argv[6], argv[7], reltol, maxiter, 1, 1);*/

	return err;
}
