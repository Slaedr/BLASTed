#include <petscksp.h>

#include <sys/time.h>
#include <time.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "perftesting.hpp"

#define PETSCOPTION_STR_LEN 30
#define PATH_STR_LEN 300

#define TASSERT(x) if(!(x)) return -1

using namespace blasted;

int main(int argc, char* argv[])
{
	char help[] = "This program solves a linear system using 2 different solvers\n\
		and optionally compares the results.\n\
		Arguments: (1) Matrix file in COO format (2) RHS file (3) Exact soln file\n\
		Use -options_file to provide a PETSc options file.\n\
		Use '-test_type compare_its' to compare the two solvers by iteration count.\n";

	if(argc < 4) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	const char *const matfile = argv[1];
	const char *const bfile = argv[2];
	const char *const xfile = argv[3];
	
	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;
	
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
	ierr = PetscOptionsGetReal(NULL, NULL, "-error_tolerance_factor", &error_tol, &set);
	if(!set) {
		printf("Error tolerance factor not set; using the default 100.");
		error_tol = 0.01;
	}

	char testtype[PETSCOPTION_STR_LEN];
	ierr = PetscOptionsGetString(NULL,NULL,"-test_type",testtype, PETSCOPTION_STR_LEN, &set);
	CHKERRQ(ierr);
	if(!set) {
		printf("Test type not set; testing convergence only.\n");
		strcpy(testtype,"convergence");
	}

	const TestParams tparams = getTestParams();

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

	PetscViewerDestroy(&matreader);
	PetscViewerDestroy(&bvecreader);
	PetscViewerDestroy(&xvecreader);

	PetscInt vs;
	ierr = VecGetSize(b, &vs); CHKERRQ(ierr);
	printf(" Rank %d: RHS size = %d.\n", rank, vs);

	std::ofstream report;
	report.open(tparams.reportfile, std::ofstream::out);

	writeHeaderToFile(report, field_width);

	TimingData refdata;

	// compute reference solution using a preconditioner from PETSc
	{
		RunParams rp;
		rp.ref = true;
		rp.nbswps = tparams.nrefbswps;
		rp.naswps = tparams.nrefaswps;
		rp.numthreads = tparams.refthreads;
		rp.nrepeats = tparams.refnruns;
		TimingData dummydata;

		run_one_test(rp, dummydata, A, b, u, refdata, report);
	}

	// run the solve to be tested as many times as requested

	for(size_t isetting = 0; isetting < tparams.threadslist.size(); isetting++)
	{
		RunParams rp;
		rp.ref = false;
		rp.nbswps = tparams.nbswps;
		rp.naswps = tparams.naswps;
		rp.numthreads = tparams.threadslist[isetting];
		rp.nrepeats = tparams.nruns;
		TimingData curtime;

		run_one_test(rp, refdata, A, b, u, curtime, report);
	}

	report.close();

	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	PetscFinalize();

	return ierr;
}
	
