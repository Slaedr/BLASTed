#undef NDEBUG

#include <petscksp.h>

#include <sys/time.h>
#include <time.h>
#include <float.h>
#include <assert.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <blasted_petsc.h>
#include "testutils.h"

#define PETSCOPTION_STR_LEN 30

#define TASSERT(x) if(!(x)) return -1

PetscReal compute_error(const MPI_Comm comm, const Vec u, const Vec uexact) {
	PetscReal errnorm;
	Vec err;
	VecDuplicate(u, &err);
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	VecNorm(err, NORM_2, &errnorm);
	VecDestroy(&err);
	return errnorm;
}

int main(int argc, char* argv[])
{
	char help[] = "This program solves a linear system.\n\
		Arguments: (1) Matrix file in COO format (2) RHS file (3) Exact soln file\n\
		Optionally, (4) whether to use the C or C++ function for the test.\n\
		Additionally, use -options_file to provide a PETSc options file.\n";

	if(argc < 4) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	char * matfile = argv[1];
	char * bfile = argv[2];
	char * xfile = argv[3];

	char *language = NULL;
	if(argc >= 5)
		language = argv[4];
	else
		language = "c";

	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help);
	if(ierr) {
		printf("Could not initialize PETSc!\n");
		fflush(stdout);
		return -1;
	}

	MPI_Comm comm = PETSC_COMM_WORLD;

	PetscMPIInt size, rank;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	DiscreteLinearProblem lp;
	ierr = readLinearSystemFromFiles(matfile, bfile, xfile, &lp); CHKERRQ(ierr);

	if(!strcmp(language,"c++")) {
		ierr = runComparisonVsPetsc_cpp(lp); CHKERRQ(ierr);
	} else {
		ierr = runComparisonVsPetsc(lp); CHKERRQ(ierr);
	}

	ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return ierr;
}

