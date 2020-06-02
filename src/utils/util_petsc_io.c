#include <petscmat.h>

#include <assert.h>
#include <string.h>

#include <utils/blasted_petsc_io.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PETSCOPTION_STR_LEN 500

int main(int argc, char *argv[])
{
	char help[] = "Writes a matrix and 2 vectors in PETSc's binary format.\n\
		Arguments: (1) Matrix file in mtx COO format (2) RHS file (mtx),\n\
        and optionally (3) -exact_soln <exact solution file>\n\
		Additionally, use -options_file to provide a PETSc options file.\n";

	if(argc < 3) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	const char *const matfile = argv[1];
	const char *const bfile = argv[2];
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

	PetscBool xset = PETSC_FALSE;
	char xfile[PETSCOPTION_STR_LEN];
	ierr = PetscOptionsGetString(NULL,NULL,"-exact_soln",xfile,PETSCOPTION_STR_LEN,&xset); CHKERRQ(ierr);

	printf(" Reading matrix and vector(s)..\n"); fflush(stdout);
	Mat A;
	Vec uexact, b;
	ierr = readMatFromCOOFile_viaSR(matfile,comm,&A); CHKERRQ(ierr);
	ierr = readVecFromFile(bfile,comm,A, &b); CHKERRQ(ierr);
	if(xset) {
		ierr = readVecFromFile(xfile,comm,A, &uexact); CHKERRQ(ierr);
	}
	printf(" Done reading matrix and vector(s).\n"); fflush(stdout);

	char outmatfile[PETSCOPTION_STR_LEN], outbfile[PETSCOPTION_STR_LEN], outxfile[PETSCOPTION_STR_LEN];

	strcpy(outmatfile, matfile);
	strcat(outmatfile, ".pmat");
	strcpy(outbfile, bfile);
	strcat(outbfile, ".pmat");
	if(xset) {
		strcpy(outxfile, xfile);
		strcat(outxfile, ".pmat");
	}

	PetscViewer matwriter;
	ierr = PetscViewerBinaryOpen(comm, outmatfile, FILE_MODE_WRITE, &matwriter); CHKERRQ(ierr);
	PetscViewer bvecwriter;
	ierr = PetscViewerBinaryOpen(comm, outbfile, FILE_MODE_WRITE, &bvecwriter); CHKERRQ(ierr);
	PetscViewer xvecwriter;
	if(xset) {
		ierr = PetscViewerBinaryOpen(comm, outxfile, FILE_MODE_WRITE, &xvecwriter); CHKERRQ(ierr);
	}

	ierr = MatView(A, matwriter); CHKERRQ(ierr);
	ierr = VecView(b, bvecwriter); CHKERRQ(ierr);
	if(xset) {
		ierr = VecView(uexact, xvecwriter); CHKERRQ(ierr);
	}

	ierr = PetscViewerDestroy(&matwriter); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&bvecwriter); CHKERRQ(ierr);
	if(xset) {
		ierr = PetscViewerDestroy(&xvecwriter); CHKERRQ(ierr);
	}

	if(xset) {
		ierr = VecDestroy(&uexact); CHKERRQ(ierr);
	}
	VecDestroy(&b);
	MatDestroy(&A);
	PetscFinalize();

	return ierr;
}

