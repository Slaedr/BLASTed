#include <petscmat.h>

#include <assert.h>
#include <string.h>

#include <blasted_petsc_io.h>

int main(int argc, char *argv[])
{
	char help[] = "Writes a matrix and 2 vectors in PETSc's binary format.\n\
		Arguments: (1) Matrix file in mtx COO format (2) RHS file (mtx) (3) Exact soln file\
		Additionally, use -options_file to provide a PETSc options file.\n";

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

	PetscViewer matwriter;
	PetscViewerBinaryOpen(comm, matfile, FILE_MODE_WRITE, &matwriter);
	PetscViewer bvecwriter;
	PetscViewerBinaryOpen(comm, bfile, FILE_MODE_WRITE, &bvecwriter);
	PetscViewer xvecwriter;
	PetscViewerBinaryOpen(comm, xfile, FILE_MODE_WRITE, &xvecwriter);

	Mat A;
	Vec uexact, b;
	ierr = readMatFromCOOFile(matfile,comm,&A); CHKERRQ(ierr);
	ierr = readVecFromFile(bfile,comm,A, &b); CHKERRQ(ierr);
	ierr = readVecFromFile(xfile,comm,A, &uexact); CHKERRQ(ierr);

	MatView(A, matwriter);
	VecView(b, bvecwriter);
	VecView(uexact, xvecwriter);

	PetscViewerDestroy(&matwriter);
	PetscViewerDestroy(&bvecwriter);
	PetscViewerDestroy(&xvecwriter);
	
	VecDestroy(&uexact);
	VecDestroy(&b);
	MatDestroy(&A);
	PetscFinalize();

	return ierr;
}
	
