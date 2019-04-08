/** \file blasted_petsc_io.cpp
 * \brief Implementation of some I/O operations involving PETSc objects
 * \author Aditya Kashi
 */

#include <cstring>
#include <vector>

#include <blasted_petsc_io.h>
#include <coomatrix.hpp>
#include <bsr/blockmatrices.hpp>

extern "C" {

PetscErrorCode readVecFromFile(const char *const file, MPI_Comm comm, Mat a, Vec *const v)
{
	PetscErrorCode ierr = 0;

	std::vector<PetscReal> array = blasted::readDenseMatrixMarket<PetscReal>(file);
	const PetscInt size = static_cast<PetscInt>(array.size());

	if(a) {
		ierr = MatCreateVecs(a, NULL, v); CHKERRQ(ierr);
	}
	else {
		ierr = VecCreate(comm, v); CHKERRQ(ierr);
		ierr = VecSetSizes(*v, PETSC_DECIDE, size); CHKERRQ(ierr);
		ierr = VecSetUp(*v); CHKERRQ(ierr);
	}

	std::vector<PetscInt> indices(size);
	for(PetscInt i = 0; i < size; i++)
		indices[i] = i;

	ierr = VecSetValues(*v, size, indices.data(), array.data(), INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = VecAssemblyBegin(*v); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(*v); CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode readMatFromCOOFile(const char *const file, MPI_Comm comm, Mat *const A)
{
	PetscErrorCode ierr = 0;
	PetscMPIInt rank, mpisize; 
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &mpisize);
	if(rank == 0)
		printf(" Num procs = %d\n", mpisize);

	blasted::COOMatrix<PetscReal,PetscInt> cmat;
	cmat.readMatrixMarket(file);
	if(rank == 0) 
		printf(" Dim of matrix is %d x %d\n", cmat.numrows(), cmat.numcols());

	ierr = MatCreate(comm, A); CHKERRQ(ierr);
	// set matrix type (and block size) from options
	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);
	PetscInt bs;
	ierr = MatGetBlockSize(*A,&bs); CHKERRQ(ierr);
	
	ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, cmat.numrows(), cmat.numcols()); CHKERRQ(ierr);

	// for default pre-allocation
	ierr = MatSetUp(*A); CHKERRQ(ierr);

	blasted::RawBSRMatrix<PetscReal,PetscInt> rmat;
	switch(bs) {
		case 3:
			cmat.convertToBSR<3,Eigen::ColMajor>(&rmat);
			break;
		case 4:
			cmat.convertToBSR<4,Eigen::ColMajor>(&rmat);
			break;
		case 5:
			cmat.convertToBSR<5,Eigen::ColMajor>(&rmat);
			break;
		case 7:
			cmat.convertToBSR<7,Eigen::ColMajor>(&rmat);
			break;
		default:
			printf("readMatFromCOOFile: Block size %d is not supported; falling back to CSR\n", bs);
			cmat.convertToCSR(&rmat);
			bs = 1;
			ierr = MatSetBlockSize(*A, bs); CHKERRQ(ierr);
	}

	ierr = MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE); CHKERRQ(ierr);
	for(PetscInt i = 0; i < rmat.nbrows; i++) {
		const PetscInt *const colinds = &rmat.bcolind[rmat.browptr[i]];
		const PetscReal *const values = &rmat.vals[rmat.browptr[i]*bs*bs];
		const PetscInt ncols = rmat.browptr[i+1]-rmat.browptr[i];
		ierr = MatSetValuesBlocked(*A, 1, &i, ncols, colinds, values, INSERT_VALUES);
		CHKERRQ(ierr);
	}
	blasted::destroyRawBSRMatrix(rmat);

	ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}

}
