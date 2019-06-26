#undef NDEBUG

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#include "../testutils.hpp"
#include "testsai.hpp"

int test_fullmatrix_interior(const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                             const LeftSAIPattern<int>& sp, const PetscInt testrow)
{
	assert(sp.nVars[testrow] == 7);
	assert(sp.nEqns[testrow] == 33);

	const int start = sp.sairowptr[testrow], end = sp.sairowptr[testrow+1];
	for(int jcol = start; jcol < end; jcol++)
	{
		const int colstart = sp.bcolptr[jcol], colend = sp.bcolptr[jcol+1];
		// TODO: test row indices in the small LHS matrix and the positions in the original matrix
	}
}

int test_fullmatrix_boundaryface(const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                                 const LeftSAIPattern<int>& sp, const PetscInt testrow)
{
}

int test_sai(const bool fullsai, const CartMesh& m, const Mat A)
{
	int ierr = 0;
	const CRawBSRMatrix<PetscScalar,PetscInt> mat = wrapLocalPetscMat(A, 1);

	const LeftSAIPattern<int> sp = left_SAI_pattern(mat);

	// Select some interior point
	const PetscInt testpoint = {3,5,4};
	assert(m.gnpoind(0) >= 5);
	assert(m.gnpoind(1) >= 7);
	assert(m.gnpoind(2) >= 6);

	const PetscInt testrow = testpoint[2]*(m.gnpoind(0)-1)*(m.gnpoind(1)-1)
		+ testpoint[1]*(m.gnpoind(0)-1) + testpoint[0];

	// Test full matrix
	ierr = test_fullmatrix_interior(mat, sp, testrow);

	return ierr;
}

