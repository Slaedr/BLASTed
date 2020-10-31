
#include <vector>
#include <stdexcept>
#include <petscmat.h>

#define PETSCOPTION_STR_LEN 500

struct COOVal {
	int row;
	int col;
	PetscScalar val;
};

int writeMatToMtx(Mat A, FILE *fout)
{
	int ierr = 0;
	PetscMPIInt size;
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	if(size > 1)
		throw std::runtime_error("Only 1 rank supported!");

	int bs;
	ierr = MatGetBlockSize(A, &bs); CHKERRQ(ierr);
	int nrows, ncols;
	ierr = MatGetSize(A, &nrows, &ncols); CHKERRQ(ierr);
	if(nrows != ncols)
		throw std::runtime_error("Must be square!");
	printf(" Dimension of matrix = %d.\n", nrows);

	int nnz;
	{
		PetscBool done = PETSC_FALSE;
		PetscInt nbrows; const PetscInt *ia;
		ierr = MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nbrows, &ia, NULL, &done); CHKERRQ(ierr);
		nnz = ia[nbrows];
		ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &nbrows, &ia, NULL, &done); CHKERRQ(ierr);
		printf(" Number of matrix nonzeros = %d.\n", nnz);
	}

	// Convert to CSR
	// Mat cA;
	// ierr = MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &cA); CHKERRQ(ierr);

	// {
	// 	PetscBool done = PETSC_FALSE;
	// 	PetscInt nbrows; const PetscInt *ia;
	// 	ierr = MatGetRowIJ(cA, 0, PETSC_FALSE, PETSC_FALSE, &nbrows, &ia, NULL, &done); CHKERRQ(ierr);
	// 	const int nnzcsr = ia[nbrows];
	// 	ierr = MatRestoreRowIJ(cA, 0, PETSC_FALSE, PETSC_FALSE, &nbrows, &ia, NULL, &done); CHKERRQ(ierr);
	// 	printf(" Petsc converted number of matrix nonzeros = %d.\n", nnzcsr);
	// 	if(nnz != nnzcsr)
	// 		throw std::runtime_error("PETSc's mat conversion failed!");
	// }

	//cmat.resize(nnz);
	fprintf(fout, "%%%%MatrixMarket matrix coordinate real general\n");
	fprintf(fout, "%d %d %d\n", nrows, nrows, nnz);

	int inz = 0;
	for(int irow = 0; irow < nrows; irow++)
	{
		const PetscScalar *vals;
		PetscInt ncols;
		const PetscInt *cols;
		ierr = MatGetRow(A, irow, &ncols, &cols, &vals); CHKERRQ(ierr);

		for(int j = 0; j < ncols; j++)
		{
			fprintf(fout, "%d %d %16.16lf\n", irow+1, cols[j]+1, vals[j]);
			inz++;
		}

		ierr = MatRestoreRow(A, irow, &ncols, &cols, &vals); CHKERRQ(ierr);
	}

	//ierr = MatDestroy(&cA); CHKERRQ(ierr);
	printf(" Wrote %d matrix entries.\n", inz);

	if(inz != nnz)
		throw std::runtime_error("Could not write correctlty: nnz = " + std::to_string(nnz) + " but wrote "
		                         + std::to_string(inz) + " entries!");

	return ierr;
}

int writeVecToMtx(Vec xvec, FILE *fout)
{
	int ierr = 0;
	PetscMPIInt size;
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	if(size > 1)
		throw std::runtime_error("Only 1 rank supported!");

	int length;
	ierr = VecGetSize(xvec, &length); CHKERRQ(ierr);
	printf(" Dimension of vector = %d.\n", length);

	fprintf(fout, "%%%%MatrixMarket matrix array real general\n");
	fprintf(fout, "%d %d\n", length, 1);

	const PetscScalar *vals;
	ierr = VecGetArrayRead(xvec, &vals); CHKERRQ(ierr);

	for(int irow = 0; irow < length; irow++)
	{
		fprintf(fout, " %16.16lf\n", vals[irow]);
	}

	ierr = VecRestoreArrayRead(xvec, &vals); CHKERRQ(ierr);
	printf(" Wrote vector.\n");
	return ierr;
}

int main(int argc, char *argv[])
{
	char help[] = "This program converts a PETSc binary matrix and vector to MTX\n\
		Arguments: (1) Matrix file in COO format (2) RHS file (3) Exact soln file\n\
		Use -options_file to provide a PETSc options file.\n";

	if(argc < 3) {
		printf("Please specify the required files.\n");
		printf("%s", help);
		return 0;
	}

	const char *const matfile = argv[1];
	const char *const bfile = argv[2];
	//const char *const xfile = argv[3];

	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;

	ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

	PetscViewer matreader;
	PetscViewerBinaryOpen(comm, matfile, FILE_MODE_READ, &matreader);
	PetscViewer bvecreader;
	PetscViewerBinaryOpen(comm, bfile, FILE_MODE_READ, &bvecreader);
	// PetscViewer xvecreader;
	// PetscViewerBinaryOpen(comm, xfile, FILE_MODE_READ, &xvecreader);

	Mat A;
	ierr = MatCreate(comm,&A); CHKERRQ(ierr);
	ierr = MatSetFromOptions(A); CHKERRQ(ierr);
	ierr = MatLoad(A, matreader); CHKERRQ(ierr);

	Vec b;
	ierr = VecCreate(comm,&b); CHKERRQ(ierr);
	//ierr = VecCreate(comm,&uexact); CHKERRQ(ierr);
	ierr = VecLoad(b, bvecreader); CHKERRQ(ierr);
	//ierr = VecLoad(uexact, xvecreader); CHKERRQ(ierr);
	//ierr = MatCreateVecs(A, &err, NULL); CHKERRQ(ierr);
	//ierr = MatCreateVecs(A, &u, NULL); CHKERRQ(ierr);

	PetscViewerDestroy(&matreader);
	PetscViewerDestroy(&bvecreader);
	// PetscViewerDestroy(&xvecreader);

	PetscInt vs;
	ierr = VecGetSize(b, &vs); CHKERRQ(ierr);
	printf(" Rank %d: RHS size = %d.\n", rank, vs);

	char outmatfile[PETSCOPTION_STR_LEN], outbfile[PETSCOPTION_STR_LEN];
	strcpy(outmatfile, matfile);
	strcat(outmatfile, ".mtx");
	strcpy(outbfile, bfile);
	strcat(outbfile, ".mtx");

	FILE *matout = fopen(outmatfile, "w");
	ierr = writeMatToMtx(A, matout); CHKERRQ(ierr);
	fclose(matout);
	FILE *vecout = fopen(outbfile, "w");
	ierr = writeVecToMtx(b, vecout); CHKERRQ(ierr);
	fclose(vecout);

	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr);

	return 0;
}
