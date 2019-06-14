/** \file
 * \brief Implementation of some testing utilities
 */

#include <stdexcept>
#include <string>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include "utils/mpiutils.hpp"
#include "utils/cmdoptions.hpp"
#include "testutils.hpp"

namespace blasted {

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

namespace test {

int compareSolverWithPetsc(const int refkspiters, const int avgkspiters,
                           Vec uref, Vec u)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters);
	fflush(stdout);

	const std::string testtype = parsePetscCmd_string("-test_type", PETSCOPTION_STR_LEN);
	const double error_tol = parseOptionalPetscCmd_real("-error_tolerance", 1e-8);
	//const double iters_tol = parseOptionalPetscCmd_real("-iters_tolerance", 1e-2);

	if(testtype == "compare_its" || testtype == "issame") {
		assert(fabs((double)refkspiters - avgkspiters)/refkspiters <= error_tol);
	}
	else if(testtype == "upper_bound_its") {
		assert(refkspiters > avgkspiters);
	}

	Vec diff;
	int ierr = VecDuplicate(u, &diff); CHKERRQ(ierr);
	ierr = VecWAXPY(diff, -1.0, u, uref); CHKERRQ(ierr);
	PetscScalar diffnorm, refnorm;
	ierr = VecNorm(uref, NORM_2, &refnorm); CHKERRQ(ierr);
	ierr = VecNorm(diff, NORM_2, &diffnorm); CHKERRQ(ierr);

	printf("Difference in solutions = %.16f.\n", diffnorm);
	printf("Relative difference = %.16f.\n", diffnorm/refnorm);
	fflush(stdout);
	if(testtype == "compare_error" || testtype == "issame")
		assert(diffnorm/refnorm <= error_tol);

	return 0;
}

}
}
