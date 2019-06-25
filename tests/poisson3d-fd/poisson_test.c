/** \file
 * \brief Executes test for comparing solvers for the Poisson problem
 */

#include "poisson_setup.h"
#include "../testutils.h"

int main(int argc, char* argv[])
{
	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n\n";

	const char *const confile = argv[1];
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

	DiscreteLinearProblem lp = setup_poisson_problem(confile);
	ierr = runComparisonVsPetsc(lp); CHKERRQ(ierr);
	ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return ierr;
}
