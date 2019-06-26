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

	DiscreteLinearProblem lp;
	ierr = readLinearSystemFromFiles(matfile, bfile, xfile, &lp); CHKERRQ(ierr);
	ierr = runComparisonVsPetsc(lp); CHKERRQ(ierr);
	ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return ierr;
}

#if 0
int main(int argc, char* argv[])
{
	char help[] = "This program solves a linear system.\n\
		Arguments: (1) Matrix file in COO format (2) RHS file (3) Exact soln file\n\
		Additionally, use -options_file to provide a PETSc options file.\n";

	if(argc < 4) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	char * matfile = argv[1];
	char * bfile = argv[2];
	char * xfile = argv[3];
	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;
	const int nruns = 1;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	// Get error check tolerance
	PetscReal error_tol;
	PetscBool set = PETSC_FALSE;
	ierr = PetscOptionsGetReal(NULL, NULL, "-error_tolerance", &error_tol, &set);
	if(!set) {
		printf("Error tolerance factor not set; using the default 100.");
		error_tol = 100.0*DBL_EPSILON;
	}

	// Error tolerance for PETSc matmult test
	PetscReal error_tol_matmult = 5e2;

	PetscViewer matreader;
	PetscViewerBinaryOpen(comm, matfile, FILE_MODE_READ, &matreader);
	PetscViewer bvecreader;
	PetscViewerBinaryOpen(comm, bfile, FILE_MODE_READ, &bvecreader);
	PetscViewer xvecreader;
	PetscViewerBinaryOpen(comm, xfile, FILE_MODE_READ, &xvecreader);

	Mat A;
	ierr = MatCreate(comm,&A); CHKERRQ(ierr);
	ierr = MatSetFromOptions(A); CHKERRQ(ierr);
	ierr = MatLoad(A, matreader); CHKERRQ(ierr);

	Vec u, uexact, b, err;
	ierr = VecCreate(comm,&b); CHKERRQ(ierr);
	ierr = VecCreate(comm,&uexact); CHKERRQ(ierr);
	ierr = VecLoad(b, bvecreader); CHKERRQ(ierr);
	ierr = VecLoad(uexact, xvecreader); CHKERRQ(ierr);
	ierr = MatCreateVecs(A, &err, NULL); CHKERRQ(ierr);
	ierr = MatCreateVecs(A, &u, NULL); CHKERRQ(ierr);
	ierr = VecSet(u, 0.0); CHKERRQ(ierr);

	PetscViewerDestroy(&matreader);
	PetscViewerDestroy(&bvecreader);
	PetscViewerDestroy(&xvecreader);
	
	// Matrix multiply check
	
	Vec test; 
	ierr = MatCreateVecs(A, NULL, &test); CHKERRQ(ierr);
	PetscInt vs; 
	VecGetLocalSize(test, &vs);
	printf(" Rank %d: Local test size = %d.\n", rank, vs);
	VecGetLocalSize(uexact, &vs);
	printf(" Rank %d: Local u_exact size = %d.\n", rank, vs);
	ierr = MatMult(A, uexact, test); CHKERRQ(ierr);
	ierr = VecAXPY(test, -1.0, b); CHKERRQ(ierr);
	PetscReal multerrnorm = 0;
	ierr = VecNorm(test, NORM_2, &multerrnorm); CHKERRQ(ierr);
	printf(" Mult error = %16.16e\n", multerrnorm);
	TASSERT(multerrnorm < error_tol_matmult*DBL_EPSILON);
	VecDestroy(&test);

	// compute reference solution using a preconditioner from PETSc

	KSP kspref; 
	ierr = KSPCreate(comm, &kspref);
	KSPSetType(kspref, KSPRICHARDSON);
	KSPRichardsonSetScale(kspref, 1.0);
	KSPSetOptionsPrefix(kspref, "ref_");
	KSPSetFromOptions(kspref);
	
	ierr = KSPSetOperators(kspref, A, A); CHKERRQ(ierr);
	
	ierr = KSPSolve(kspref, b, u); CHKERRQ(ierr);

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	PetscReal errnormref = compute_error(comm,u,uexact);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	KSPDestroy(&kspref);

	// run the solve to be tested as many times as requested

	int totalkspiters = 0;
	PetscReal errnorm = 0;
	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);
		KSP ksp;

		ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
		KSPSetType(ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(ksp, 1.0);
	
		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

		//setup_localpreconditioner_blasted(ksp, &bctx);
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		KSPGetIterationNumber(ksp, &kspiters);
		totalkspiters += kspiters;

		if(rank == 0) {
			//printf(" Number of KSP iterations = %d\n", kspiters);
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f\n", rnorm);
		}

		errnorm = compute_error(comm,u,uexact);
		if(rank == 0) {
			printf("Test run:\n");
			printf(" error and log error: %.16f, %f\n", errnorm, log10(errnorm));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		destroyBlastedDataList(&bctx);
	}

	const int avgkspiters = totalkspiters/(double)nruns;
	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters);
	fflush(stdout);

	// Test 1
	assert(avgkspiters/nruns == refkspiters);

	if (rank == 0)
		printf("Difference in error norms = %g.\n", fabs(errnorm-errnormref));
	fflush(stdout);
	// Test 2
	assert(fabs(errnorm-errnormref) < error_tol);

	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	PetscFinalize();

	return ierr;
}
#endif

