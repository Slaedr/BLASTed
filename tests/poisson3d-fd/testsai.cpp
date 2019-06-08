#undef NDEBUG

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "../testutils.hpp"
#include "testsai.hpp"

int test_sai(const bool fullsai, const CartMesh& m, const Mat A)
{
	int ierr = 0;
	const CRawBSRMatrix<PetscScalar,PetscInt> mat = wrapLocalPetscMat(A, 1);

	const TriangularLeftSAIPattern<PetscInt> tpattern = fullsai ?
		triangular_SAI_pattern(mat) : triangular_incomp_SAI_pattern(mat);

	// Select some interior point
	const PetscInt testpoint = {3,5,4};
	assert(m.gnpoind(0) >= 5);
	assert(m.gnpoind(1) >= 7);
	assert(m.gnpoind(2) >= 6);
	const PetscInt testrow = testpoint[2]*(m.gnpoind(0)-1)*(m.gnpoind(1)-1)
		+ testpoint[1]*(m.gnpoind(0)-1) + testpoint[0];

	return ierr;
}

