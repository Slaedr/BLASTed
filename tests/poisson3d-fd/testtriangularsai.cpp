#undef NDEBUG
#include <cassert>

#include "srmatrixdefs.hpp"
#include "../../src/sai.hpp"
#include "../testutils.hpp"
#include "poisson_setup.h"
#include "poisson_utils.hpp"
#include "testsai.hpp"

using namespace blasted;

void test_incomplete_uppertri_interior(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                                       const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
}

void test_incomplete_lowertri_interior(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                                       const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
	assert(testpoint[0] >= 2);
	assert(testpoint[1] >= 2);
	assert(testpoint[2] >= 2);
	assert(testpoint[0] <= m.gnpoind(0)-4);
	assert(testpoint[1] <= m.gnpoind(1)-4);
	assert(testpoint[2] <= m.gnpoind(2)-4);

	const PetscInt backpoint[] = { testpoint[0], testpoint[1], testpoint[2]-1 };
	const PetscInt backrow = getMatRowIdx(m, backpoint);
	const PetscInt downpoint[] = { testpoint[0], testpoint[1]-1, testpoint[2] };
	const PetscInt downrow = getMatRowIdx(m, downpoint);
	const PetscInt leftpoint[] = { testpoint[0]-1, testpoint[1], testpoint[2] };
	const PetscInt leftrow = getMatRowIdx(m, leftpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	assert(sp.nVars[testrow] == 4);
	assert(sp.nEqns[testrow] == 4);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == 4);

	for(int jcol = firstcol; jcol < lastcol; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
		if(jcol == firstcol+3) {
			printf(" Number of diagonal entries = %d.\n",colsize); fflush(stdout);
			assert(colsize == 4);
		}
		else
			assert(colsize == 1);

		if(jcol == firstcol) {
			// Back
			assert(sp.browind[colstart] == 0);
			assert(sp.bpos[colstart] == mat.diagind[backrow]);
		}
		else if(jcol == firstcol+1) {
			// Down
			assert(sp.browind[colstart] == 1);
			assert(sp.bpos[colstart] == mat.diagind[downrow]);
		}
		else if(jcol == firstcol+2) {
			// Left
			assert(sp.browind[colstart] == 2);
			assert(sp.bpos[colstart] == mat.diagind[leftrow]);
		}
		else if(jcol == firstcol+3) {
			// Centre
			for(int j = 0; j < 4; j++) {
				assert(sp.browind[colstart+j] == j);
				assert(sp.bpos[colstart+j] == mat.browptr[testrow]+j);
			}
		}
	}
}

int test_sai(const bool fullsai, const bool upper, const CartMesh& m, const Mat A)
{
	int ierr = 0;

	const CRawBSRMatrix<PetscScalar,PetscInt> mat = wrapLocalPetscMat(A, 1);

	const CRawBSRMatrix<PetscScalar,PetscInt> tmat = upper ?
		getUpperTriangularView(mat) : getLowerTriangularView(mat);

	// Test interior point
	{
		const PetscInt testpoint[] = {3,3,3};
		assert(m.gnpoind(0) >= 5);
		assert(m.gnpoind(1) >= 7);
		assert(m.gnpoind(2) >= 6);

		const PetscInt testrow = getMatRowIdx(m, testpoint);

		PetscInt ncols = 0;
		ierr = MatGetRow(A, testrow, &ncols, NULL, NULL); CHKERRQ(ierr);
		printf("Number of cols in row %d is %d.\n", testrow, ncols);

		if(fullsai) {
			// const LeftSAIPattern<int> sp = left_SAI_pattern(mat);
			// test_fullmatrix_interior(m, mat, sp, testpoint);
		}
		else {
			const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(tmat);
			if(upper)
				// nothing yet
			else
				test_incomplete_lowertri_interior(m, tmat, sp, testpoint);
		}
	}

	// Test +i boundary point
	{
		const PetscInt testpoint[] = {m.gnpoind(0)-2, 3, 3};
		if(fullsai) {
			const LeftSAIPattern<int> sp = left_SAI_pattern(mat);
			test_fullmatrix_boundaryface(m,mat,sp,testpoint);
		}
		else {
			// const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(mat);
		}
	}

	alignedDestroyRawBSRMatrixTriangularView(mat);

	return ierr;
}

int main(int argc, char *argv[])
{
	assert(argc > 3);

	const std::string confile = argv[1];
	const std::string test_type = argv[2];
	const std::string uplow = argv[3];

	int ierr = PetscInitialize(&argc, &argv, NULL, NULL);

	{
		const test::MeshSpec ms = test::readInputFile(PETSC_COMM_WORLD, confile.c_str());
		const CartMesh mesh = createMesh(PETSC_COMM_WORLD, ms);
		DiscreteLinearProblem lp = setup_poisson_problem(confile.c_str());

		if(test_type == "fullsai")
			if(uplow == "upper")
				ierr = test_sai(true, true, mesh, lp.lhs);
			else
				ierr = test_sai(true, false, mesh, lp.lhs);
		else
			if(uplow == "upper")
				ierr = test_sai(false, true, mesh, lp.lhs);
			else
				ierr = test_sai(false, false, mesh, lp.lhs);

		ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);
	}
	ierr = PetscFinalize();
	return ierr;
}
