/** \file testpetscsolver.cpp
 * \brief Helper functions to test preconditioners with PETSc
 * \author Aditya Kashi
 */

#include <cstring>
#include <vector>
#include "testpetscsolver.h"
#include "../src/coomatrix.hpp"
#include "../src/blockmatrices.hpp"

extern "C" {

PetscErrorCode readVecFromFile(const char *const file, Vec v)
{
	PetscErrorCode ierr = 0;

	std::vector<PetscReal> array = blasted::readDenseMatrixMarket(file);
	const PetscInt size = array.size();
	PetscInt *indices = new PetscInt[size];
	for(PetscInt i = 0; i < size; i++)
		indices[i] = i;
	for(PetscInt i = 0; i < size; i++)
		VecSetValues(v, size, indices, array, INSERT_VALUES);

	VecAssemblyBegin();
	VecAssemblyEnd();

	delete [] indices;
	return ierr;
}

PetscErrorCode readMatFromCOOFile(const char *const file, Mat A)
{
	PetscErrorCode ierr = 0;

	blasted::COOMatrix<PetscReal,PetscInt> cmat;
	cmat.readMatrixMarket(file);

	MatType mattype;
	ierr = MatGetType(A, &mattype); CHKERRQ(ierr);

	if(!strcmp(mattype,MATSEQAIJ) || !strcmp(mattype,MATMPIAIJ) || !strcmp(mattype,MATAIJ))
	{
		blasted::RawBSRMatrix<PetscReal,PetscInt> rmat;
		cmat.convertToCSR(&rmat);
		for(PetscInt i = 0; i < rmat.nbrows; i++) {
			for(PetscInt j = rmat.browptr[i]; j < rmat.browptr[i+1]; j++) 
			{
				const PetscInt *const colinds = &rmat.bcolind[rmat.browptr[i]];
				const PetscReal *const values = &rmat.vals[rmat.browptr[i]];
				const PetscInt ncols = rmat.browptr[i+1]-rmat.browptr[i];
				ierr = MatSetValues(A, 1, &i, ncols, colinds, values, INSERT_VALUES);
				CHKERRQ(ierr);
			}
		}
		blasted::destroyRawBSRMatrix(rmat);
	}
	else if(!strcmp(mattype,MATSEQBAIJ) || !strcmp(mattype,MATMPIBAIJ) || !strcmp(mattype,MATBAIJ))
	{
		blasted::RawBSRMatrix<PetscReal,PetscInt> rmat;
		PetscInt bs;
		ierr = MatGetBlockSize(A,&bs); CHKERRQ(ierr);
		switch(bs) {
			case 3:
				cmat.convertToBSR<3,ColMajor>(&rmat);
				break;
			case 4:
				cmat.convertToBSR<4,ColMajor>(&rmat);
				break;
			case 5:
				cmat.convertToBSR<5,ColMajor>(&rmat);
				break;
			case 7:
				cmat.convertToBSR<7,ColMajor>(&rmat);
				break;
			default:
				printf("! readMatFromCOOFile: Block size %d is not supported!\n", bs);
				abort();
		}

		for(PetscInt i = 0; i < rmat.nbrows; i++) {
			for(PetscInt j = rmat.browptr[i]; j < rmat.browptr[i+1]; j++) 
			{
				const PetscInt *const colinds = &rmat.bcolind[rmat.browptr[i]];
				const PetscReal *const values = &rmat.vals[rmat.browptr[i]*bs*bs];
				const PetscInt ncols = rmat.browptr[i+1]-rmat.browptr[i];
				ierr = MatSetValuesBlocked(A, 1, &i, ncols, colinds, values, INSERT_VALUES);
				CHKERRQ(ierr);
			}
		}
		blasted::destroyRawBSRMatrix(rmat);
	}
	else {
		printf("! readMatFromCOOFile: Matrix type %s not supported!\n", mattype);
		abort();
	}

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	return ierr;
}

}
