/** \file
 * \brief Implementation of some testing utilities
 */

#undef NDEBUG
#include <cassert>
#include <stdexcept>
#include <string>
#include <float.h>

#include <petscksp.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "utils/mpiutils.hpp"
#include "utils/cmdoptions.hpp"
#include "testutils.h"
#include "testutils.hpp"

#define PETSCOPTION_STR_LEN 30

namespace blasted {

SRMatrixStorage<const PetscScalar,const PetscInt> wrapLocalPetscMat(Mat A, const int bs)
{
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); petsc_throw(ierr);
	ierr = MatGetLocalSize(A, &localrows, &localcols); petsc_throw(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); petsc_throw(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	if(bs == 1) {
		const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
		assert(Adiag != NULL);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows, Adiag->i[localrows],
		                                                          Adiag->i[localrows]);
	}
	else {
		const Mat_SeqBAIJ *const Adiag = (const Mat_SeqBAIJ*)A->data;
		assert(Adiag != NULL);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows, Adiag->i[localrows],
		                                                          Adiag->i[localrows]);
	}
}

extern "C" {

int compareSolverWithRef(const int refkspiters, const int avgkspiters,
                         Vec uref, Vec u)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters);
	fflush(stdout);

	const std::string testtype = parsePetscCmd_string("-test_type", PETSCOPTION_STR_LEN);
	const double error_tol = parseOptionalPetscCmd_real("-error_tolerance", 2*DBL_EPSILON);
	//const double iters_tol = parseOptionalPetscCmd_real("-iters_tolerance", 1e-2);

	if(rank == 0)
		printf("  Test tolerance = %g.\n", error_tol);

	if(testtype == "compare_its" || testtype == "issame") {
		assert(fabs((double)refkspiters - avgkspiters)/refkspiters <= error_tol);
	}
	else if(testtype == "upper_bound_its") {
		assert(refkspiters > avgkspiters);
	}

	Vec diff;
	int ierr = VecDuplicate(u, &diff); CHKERRQ(ierr);
	ierr = VecSet(diff, 0); CHKERRQ(ierr);
	ierr = VecWAXPY(diff, -1.0, u, uref); CHKERRQ(ierr);
	PetscScalar diffnorm, refnorm;
	ierr = VecNorm(uref, NORM_2, &refnorm); CHKERRQ(ierr);
	ierr = VecNorm(diff, NORM_2, &diffnorm); CHKERRQ(ierr);
	ierr = VecDestroy(&diff); CHKERRQ(ierr);

	printf("Difference in solutions = %g.\n", diffnorm);
	printf("Relative difference = %g.\n", diffnorm/refnorm);
	fflush(stdout);
	if(testtype == "compare_error" || testtype == "issame")
		assert(diffnorm/refnorm <= error_tol);

	return 0;
}

}
}

// some unused snippets that might be useful at some point

/* Wall clock time by Unix functions

struct timeval time1, time2;
gettimeofday(&time1, NULL);
double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

some code here..

gettimeofday(&time2, NULL);
double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
*/

/* For viewing the ILU factors computed by PETSc PCILU

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/factor/factor.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>

if(precch == 'i') {	
	// view factors
	PC_ILU* ilu = (PC_ILU*)pc->data;
	//PC_Factor* pcfact = (PC_Factor*)pc->data;
	//Mat fact = pcfact->fact;
	Mat fact = ((PC_Factor*)ilu)->fact;
	printf("ILU0 factored matrix:\n");

	Mat_SeqAIJ* fseq = (Mat_SeqAIJ*)fact->data;
	for(int i = 0; i < fact->rmap->n; i++) {
		printf("Row %d: ", i);
		for(int j = fseq->i[i]; j < fseq->i[i+1]; j++)
			printf("(%d: %f) ", fseq->j[j], fseq->a[j]);
		printf("\n");
	}
}
*/

