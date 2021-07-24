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
	const auto params = blasted_testsolve::read_from_cmd(argc, argv);
	if(params.maxiter == -1)
		return 0;

	int err = 0;
	if(params.mattype == "bsr") {
		if(params.blocksize == 4)
			err = blasted_testsolve::test_solve<4>(params);
#ifdef BUILD_BLOCK_SIZE
		else if(block_size == BUILD_BLOCK_SIZE)
			err = blasted_testsolve::test_solve<BUILD_BLOCK_SIZE>(params);
#endif
		else
			throw std::runtime_error("unsupported block size!");
	}
	else
		err = blasted_testsolve::test_solve<1>(params);

	return err;
}
