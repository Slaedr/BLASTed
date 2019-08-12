/** \file
 * \brief Implementation of test for convergence of async (B)ILU factorization with sweeps
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

#include "../testutils.h"
#include "async_ilu_convergence.hpp"

DiscreteLinearProblem generateDiscreteProblem(const int argc, const char *argv[]);

int main(int argc, char *argv[])
{
	if(argc < 2) {
		printf(" ! Please provide 'poisson' or 'file'\n");
		exit(-1);
	}

	const DiscreteLinearProblem dlp = generateDiscreteProblem(argc, argv);

	int ierr = 0;

	ierr = destroyDiscreteLinearProblem(&dlp);
	return ierr;
}

DiscreteLinearProblem generateDiscreteProblem(const int argc, const char *argv[])
{
	DiscreteLinearProblem dlp;
	if(!strcmp(argv[1],"poisson"))
	{
		if(argc < 3) {
			printf(" ! Please provide a Poisson control file!\n");
			exit(-1);
		}

		dlp = setup_poisson_problem(argv[2]);
	}
	else {
		if(argc < 5) {
			printf(" ! Please provide filenames for LHS, RHS vector and exact solution (in order).\n");
			exit(-1);
		}

		int ierr = readLinearSystemFromFiles(argv[2], argv[3], argv[4], &dlp);
		assert(ierr == 0);
	}

	return dlp;
}
