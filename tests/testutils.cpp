/** \file
 * \brief Implementation of some testing utilities
 */

#include <stdexcept>
#include <string>
#include <petscerror.h>
#include "testutils.hpp"

void petsc_check(const int ierr)
{
	if(ierr) {
		std::string errmsg = "PETSc ";
		switch(ierr) {
		case PETSC_ERR_MEM:
			errmsg += "memory error!";
			break;
		case PETSC_ERR_SUP:
			errmsg += "unsupported operation!";
			break;
		default:
			errmsg += " error!";
		}
		throw std::runtime_error(errmsg);
	}
}

CRawBSRMatrix<PetscScalar,PetscInt> wrapLocalPetscMat(Mat A, const int bs)
{
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); petsc_check(ierr);
	ierr = MatGetLocalSize(A, &localrows, &localcols); petsc_check(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); petsc_check(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
	const Mat_SeqBAIJ *const Abdiag = (const Mat_SeqBAIJ*)A->data;

	CRawBSRMatrix<PetscScalar,PetscInt> rmat;
	rmat.nbrows = localrows/bs;
	if(bs == 1) {
		assert(Adiag != NULL);
		rmat.browptr = Adiag->i;
		rmat.bcolind = Adiag->j;
		rmat.diagind = Adiag->diag;
		rmat.vals = Adiag->a;
	}
	else {
		assert(Abdiag != NULL);
		rmat.browptr = Abdiag->i;
		rmat.bcolind = Abdiag->j;
		rmat.diagind = Abdiag->diag;
		rmat.vals = Abdiag->a;
	}

	return rmat;
}
