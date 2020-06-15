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
#define MAX_SWEEPS 10

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
	char help[] = "This program solves a linear system for a number of async sweeps.\n\
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

	int nbuilds = MAX_SWEEPS;
	int buildsweeps[MAX_SWEEPS];
	PetscBool set = 0;
	ierr = PetscOptionsGetIntArray(NULL, NULL, "-blasted_async_paramsweep_build", buildsweeps,
	                               &nbuilds, &set);
	if(!set) {
		printf("Please set -blasted_async_param_sweep_build to an array of integers\n");
		exit(-1);
	}

	int napplys = MAX_SWEEPS;
	int applysweeps[MAX_SWEEPS];
	set = 0;
	ierr = PetscOptionsGetIntArray(NULL, NULL, "-blasted_async_paramsweep_apply", applysweeps,
	                               &napplys, &set);
	if(!set) {
		printf("Please set -blasted_async_param_sweep_apply to an array of integers\n");
		exit(-1);
	}

	set = PETSC_FALSE;
	int cmdnumruns;
	ierr = PetscOptionsGetInt(NULL,NULL,"-num_runs",&cmdnumruns,&set); CHKERRQ(ierr);
	const int nruns = set ? cmdnumruns : 1;
	printf(" Using %d runs per sweep combination.\n", nruns);

	set = PETSC_FALSE;
	char outfilename[1000];
	ierr = PetscOptionsGetString(NULL, NULL, "-blasted_async_paramsweep_outputfile", outfilename, 1000,
	                             &set);
	if(!set) {
		printf("Please give a path for -blasted_async_paramsweep_outputfile\n");
		exit(-1);
	}

	int itertable[MAX_SWEEPS][MAX_SWEEPS];
	double devtable[MAX_SWEEPS][MAX_SWEEPS];

	DiscreteLinearProblem lp;
	ierr = readLinearSystemFromFiles(matfile, bfile, xfile, &lp, false); CHKERRQ(ierr);

	for(int ibuild = 0; ibuild < nbuilds; ibuild++)
		for(int iapply = 0; iapply < napplys; iapply++)
		{
			printf(" Setting: build %d sweeps, apply %d sweeps.\n", buildsweeps[ibuild], applysweeps[iapply]);
			int iters;
			ierr = runPetsc(lp, buildsweeps[ibuild], applysweeps[iapply], nruns, &iters,
			                &devtable[ibuild][iapply]);
			CHKERRQ(ierr);
			itertable[ibuild][iapply] = iters;
		}

	ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);

	if(rank == 0)
	{
		FILE *outf = fopen(outfilename, "w");

		fprintf(outf, "Linear system with matrix from %s\n\n", matfile);
		fprintf(outf, "Number of linear system iterations:\n");
		fprintf(outf, "Apply sweeps ->");
		for(int iapply = 0; iapply < napplys; iapply++)
			fprintf(outf, "%8d", applysweeps[iapply]);
		fprintf(outf, "\nBuild sweeps   \n----------------------\n");

		for(int ibuild = 0; ibuild < nbuilds; ibuild++)
		{
			fprintf(outf, "%12d | ", buildsweeps[ibuild]);
			for(int iapply = 0; iapply < napplys; iapply++)
				fprintf(outf, "%8d", itertable[ibuild][iapply]);
			fprintf(outf, "\n");
		}

		fprintf(outf, "\nRelative deviations:\n");
		fprintf(outf, "Apply sweeps ->");
		for(int iapply = 0; iapply < napplys; iapply++)
			fprintf(outf, "%10d", applysweeps[iapply]);
		fprintf(outf, "\nBuild sweeps   \n----------------------\n");

		for(int ibuild = 0; ibuild < nbuilds; ibuild++)
		{
			fprintf(outf, "%14d | ", buildsweeps[ibuild]);
			for(int iapply = 0; iapply < napplys; iapply++)
				fprintf(outf, "%10.3e", devtable[ibuild][iapply]);
			fprintf(outf, "\n");
		}

		fclose(outf);
	}

	ierr = PetscFinalize();
	return ierr;
}

