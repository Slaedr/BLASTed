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

#define PETSCOPTION_STR_LEN 200

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
		Arguments: (1) Matrix file in PETSc binary format (2) RHS file (3) Exact soln file\n\
		Additionally, use -options_file to provide a PETSc options file.\n";

	if(argc < 3) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help);
	if(ierr) {
		printf("Could not initialize PETSc!\n");
		fflush(stdout);
		return -1;
	}

	const char *const matfile = argv[1];
	const char *const bfile = argv[2];

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

	PetscBool set = PETSC_FALSE;
	int cmdnumruns;
	ierr = PetscOptionsGetInt(NULL,NULL,"-num_runs",&cmdnumruns,&set); CHKERRQ(ierr);
	const int nruns = set ? cmdnumruns : 1;
	printf(" Using %d runs.\n", nruns);

	char *xfile;
	char solnfile[PETSCOPTION_STR_LEN];
	ierr = PetscOptionsGetString(NULL,NULL,"-solution_file", solnfile, PETSCOPTION_STR_LEN, &set);
	CHKERRQ(ierr);
	if(set)
		xfile = solnfile;
	else
		xfile = NULL;

	DiscreteLinearProblem lp;
	ierr = readLinearSystemFromFiles(matfile, bfile, xfile, &lp, true); CHKERRQ(ierr);

	double avgkspiters = 0;
	int *const runiters = malloc(nruns*sizeof(int));
	int irun = 0;
	for( ; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);

		Vec urun;
		ierr = VecDuplicate(lp.b, &urun); CHKERRQ(ierr);   // Assuming square system

		KSP ksp;

		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);

		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, lp.lhs, lp.lhs); CHKERRQ(ierr);

		// Create BLASTed data structure and setup the PC
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, lp.b, urun); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		ierr = KSPGetIterationNumber(ksp, &kspiters); CHKERRQ(ierr);
		avgkspiters += kspiters;
		runiters[irun] = kspiters;

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		if(rank == 0)
			printf("  KSP converged reason = %d.\n", ksp_reason);
		//assert(ksp_reason > 0);
		if(ksp_reason <= 0) {
			if(rank == 0)
				printf("KSP did not converge!\n");
			avgkspiters = -100;
			irun++;
			break;
		}

		if(rank == 0) {
			ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
			printf(" KSP residual norm = %f\n", rnorm);
		}

		//errnorm += compute_error(comm,m,da,u,lp.uexact);
		/* PetscReal errnormrun = 0; */
		/* ierr = compute_difference_norm(u, lp.uexact, &errnormrun); CHKERRQ(ierr); */

		//ierr = VecAXPY(u, 1.0, urun); CHKERRQ(ierr);

		/* if(rank == 0) { */
		/* 	printf("Test run:\n"); */
		/* 	printf(" error: %.16f\n", errnormrun); */
		/* 	printf(" log error: %f\n", log10(errnormrun)); */
		/* } */

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		ierr = VecDestroy(&urun); CHKERRQ(ierr);

		//computeTotalTimes(&bctx);
		destroyBlastedDataList(&bctx);
	}

	avgkspiters = avgkspiters/(double)nruns;
	//ierr = VecScale(u, 1.0/nruns); CHKERRQ(ierr);
	if(rank == 0)
		printf("Average iteration count = %g\n", avgkspiters);
	if(avgkspiters < 0) {
		if(rank == 0)
			printf("IMPORTANT: One of the runs diverged\n");
	}
	else
		assert(irun == nruns);

	double reldev = 0;
	for(int i = 0; i < irun; i++)
		reldev += (runiters[i]-avgkspiters)*(runiters[i]-avgkspiters);
	reldev = sqrt(reldev/irun)/avgkspiters;

	printf("Relative deviation in iterations = %f\n", reldev);

	free(runiters);

	ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return ierr;
}

