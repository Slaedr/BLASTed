#undef NDEBUG

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#include "srmatrixdefs.hpp"
#include "../../src/sai.hpp"
#include "../testutils.hpp"
#include "poisson_setup.h"
#include "poisson_utils.hpp"
#include "testsai.hpp"

using namespace blasted;

static inline PetscInt getMatRowIdx(const CartMesh& m, const PetscInt testpoint[3])
{
	return (testpoint[2]-1)*(m.gnpoind(0)-2)*(m.gnpoind(1)-2)
		+ (testpoint[1]-1)*(m.gnpoind(0)-2) + testpoint[0]-1;
}

void test_incomplete_fullmatrix_interior(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
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
	const PetscInt rightpoint[] = { testpoint[0]+1, testpoint[1], testpoint[2] };
	const PetscInt rightrow = getMatRowIdx(m, rightpoint);
	const PetscInt uppoint[] = { testpoint[0], testpoint[1]+1, testpoint[2] };
	const PetscInt uprow = getMatRowIdx(m, uppoint);
	const PetscInt frontpoint[] = { testpoint[0], testpoint[1], testpoint[2]+1 };
	const PetscInt frontrow = getMatRowIdx(m, frontpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	assert(sp.nVars[testrow] == 7);
	assert(sp.nEqns[testrow] == 7);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == 7);

	for(int jcol = firstcol; jcol < lastcol; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
		if(jcol == firstcol+3) {
			printf(" Number of diagonal entries = %d.\n",colsize); fflush(stdout);
			assert(colsize == 7);
		}
		else
			assert(colsize == 2);

		if(jcol == firstcol) {
			// Back
			assert(sp.browind[colstart] == 0);
			assert(sp.bpos[colstart] == mat.diagind[backrow]);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.bpos[colstart+1] == mat.browptr[backrow]+6);
		}
		else if(jcol == firstcol+1) {
			assert(sp.browind[colstart] == 1);
			assert(sp.bpos[colstart] == mat.diagind[downrow]);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.bpos[colstart+1] == mat.browptr[downrow]+5);
		}
		else if(jcol == firstcol+2) {
			assert(sp.browind[colstart] == 2);
			assert(sp.bpos[colstart] == mat.diagind[leftrow]);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.bpos[colstart+1] == mat.browptr[leftrow]+4);
		}
		else if(jcol == firstcol+3) {
			for(int j = 0; j < 7; j++) {
				assert(sp.browind[colstart+j] == j);
				assert(sp.bpos[colstart+j] == mat.browptr[testrow]+j);
			}
		}
		else if(jcol == firstcol+4) {
			assert(sp.browind[colstart] == 3);
			assert(sp.bpos[colstart] == mat.browptr[rightrow]+2);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.bpos[colstart+1] == mat.diagind[rightrow]);
		}
		else if(jcol == firstcol+5) {
			assert(sp.browind[colstart] == 3);
			assert(sp.bpos[colstart] == mat.browptr[uprow]+1);
			assert(sp.browind[colstart+1] == 5);
			assert(sp.bpos[colstart+1] == mat.diagind[uprow]);
		}
		else if(jcol == firstcol+6) {
			assert(sp.browind[colstart] == 3);
			assert(sp.bpos[colstart] == mat.browptr[frontrow]);
			assert(sp.browind[colstart+1] == 6);
			assert(sp.bpos[colstart+1] == mat.diagind[frontrow]);
		}
	}
}

void test_fullmatrix_interior(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
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
	const PetscInt rightpoint[] = { testpoint[0]+1, testpoint[1], testpoint[2] };
	const PetscInt rightrow = getMatRowIdx(m, rightpoint);
	const PetscInt uppoint[] = { testpoint[0], testpoint[1]+1, testpoint[2] };
	const PetscInt uprow = getMatRowIdx(m, uppoint);
	const PetscInt frontpoint[] = { testpoint[0], testpoint[1], testpoint[2]+1 };
	const PetscInt frontrow = getMatRowIdx(m, frontpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	assert(sp.nVars[testrow] == 7);
	assert(sp.nEqns[testrow] == 25);

	const int start = sp.sairowptr[testrow], end = sp.sairowptr[testrow+1];
	for(int jcol = start; jcol < end; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		for(int i = 0; i < 7; i++) {
			assert(sp.browind[colstart+i] >= 0);
			assert(sp.browind[colstart+i] < sp.nEqns[testrow]);
		}

		if(jcol == start) {
			// back column
			for(int i = 0; i < 6; i++) {
				assert(sp.browind[colstart+i] == i);
			}
			assert(sp.browind[colstart+6] == 12);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[backrow]+i);
		}
		else if(jcol == start+1) {
			// down column
			assert(sp.browind[colstart] == 1);
			for(int i = 1; i < 5; i++) {
				assert(sp.browind[colstart+i] == 5+i);
			}
			assert(sp.browind[colstart+5] == 12);
			assert(sp.browind[colstart+6] == 19);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[downrow]+i);
		}
		else if(jcol == start+2) {
			// left column
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 7);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 8+i);
			assert(sp.browind[colstart+5] == 15);
			assert(sp.browind[colstart+6] == 20);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[leftrow]+i);
		}
		else if(jcol == start+3) {
			// centre column
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 8);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 9+i);
			assert(sp.browind[colstart+5] == 16);
			assert(sp.browind[colstart+6] == 21);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[testrow]+i);
		}
		else if(jcol == start+4) {
			// right column
			assert(sp.browind[colstart] == 4);
			assert(sp.browind[colstart+1] == 9);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 10+i);
			assert(sp.browind[colstart+5] == 17);
			assert(sp.browind[colstart+6] == 22);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[rightrow]+i);
		}
		else if(jcol == start+5) {
			// up column
			assert(sp.browind[colstart] == 5);
			assert(sp.browind[colstart+1] == 12);
			for(int i = 2; i < 6; i++)
				assert(sp.browind[colstart+i] == 13+i);
			assert(sp.browind[colstart+6] == 23);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[uprow]+i);
		}
		else if(jcol == start+6) {
			// front column
			assert(sp.browind[colstart] == 12);
			for(int i = 1; i < 7; i++)
				assert(sp.browind[colstart+i] == 18+i);

			for(int i = 0; i < 7; i++)
				assert(sp.bpos[colstart+i] == mat.browptr[frontrow]+i);
		}
		else
			throw std::runtime_error("Invalid colummn!");
	}
}

void test_fullmatrix_boundaryface(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                                  const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
	// let's do the +i face
	assert(testpoint[0] == m.gnpoind(0)-2);

	assert(testpoint[1] >= 3);
	assert(testpoint[2] >= 3);
	assert(testpoint[1] <= m.gnpoind(1)-4);
	assert(testpoint[2] <= m.gnpoind(2)-4);

	const PetscInt testrow = getMatRowIdx(m,testpoint);

	assert(sp.nVars[testrow] == 6);
	assert(sp.nEqns[testrow] == 19);

	const int start = sp.sairowptr[testrow], end = sp.sairowptr[testrow+1];
	assert(end-start == 6);
	for(int jcol = start; jcol < end; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		if(jcol == start+2) {
			printf(" Size of col = %d.\n", end-start); fflush(stdout);
			for(int i = 0; i < 7; i++) {
				assert(sp.browind[colstart+i] >= 0);
				assert(sp.browind[colstart+i] < sp.nEqns[testrow]);
			}
		}
		else {
			for(int i = 0; i < 6; i++) {
				assert(sp.browind[colstart+i] >= 0);
				assert(sp.browind[colstart+i] < sp.nEqns[testrow]);
			}
		}

		if(jcol == start) {
			// back column
			for(int i = 0; i < 5; i++) {
				//printf("  LHS row ind = %d. ", sp.browind[colstart+i]); fflush(stdout);
				assert(sp.browind[colstart+i] == i);
			}
			printf(" Colstart +5 = %d.\n", sp.browind[colstart+5]); fflush(stdout);
			assert(sp.browind[colstart+5] == 10);
		}
		else if(jcol == start+1) {
			// down column
			assert(sp.browind[colstart] == 1);
			for(int i = 1; i < 4; i++)
				assert(sp.browind[colstart+i] == 4+i);
			assert(sp.browind[colstart+4] == 10);
			assert(sp.browind[colstart+5] == 14);
		}
		else if(jcol == start+2) {
			// left column
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 6);
			for(int i = 2; i < 6; i++)
				assert(sp.browind[colstart+i] == 6+i);
			assert(sp.browind[colstart+6] == 15);
		}
		else if(jcol == start+3) {
			// centre column
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 7);
			assert(sp.browind[colstart+2] == 9);
			assert(sp.browind[colstart+3] == 10);
			assert(sp.browind[colstart+4] == 12);
			assert(sp.browind[colstart+5] == 16);
		}
		// right column does not exist
		else if(jcol == start+4) {
			// up column
			assert(sp.browind[colstart] == 4);
			for(int i = 1; i < 5; i++)
				assert(sp.browind[colstart+i] == 9+i);
			assert(sp.browind[colstart+5] == 17);
		}
		else if(jcol == start+5) {
			// front column
			assert(sp.browind[colstart] == 10);
			for(int i = 1; i < 6; i++)
				assert(sp.browind[colstart+i] == 13+i);
		}
		else
			throw std::runtime_error("Invalid colummn!");
	}
}

int test_sai(const bool fullsai, const CartMesh& m, const Mat A)
{
	int ierr = 0;

	const CRawBSRMatrix<PetscScalar,PetscInt> mat = wrapLocalPetscMat(A, 1);

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

		// Test full matrix
		if(fullsai) {
			const LeftSAIPattern<int> sp = left_SAI_pattern(mat);
			test_fullmatrix_interior(m, mat, sp, testpoint);
		}
		else {
			const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(mat);
			test_incomplete_fullmatrix_interior(m, mat, sp, testpoint);
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

	return ierr;
}

int main(int argc, char *argv[])
{
	const std::string confile = argv[1];
	const std::string test_type = argv[2];

	int ierr = PetscInitialize(&argc, &argv, NULL, NULL);

	{
		const test::MeshSpec ms = test::readInputFile(PETSC_COMM_WORLD, confile.c_str());
		const CartMesh mesh = createMesh(PETSC_COMM_WORLD, ms);
		DiscreteLinearProblem lp = setup_poisson_problem(confile.c_str());

		if(test_type == "fullsai")
			ierr = test_sai(true, mesh, lp.lhs);
		else
			ierr = test_sai(false, mesh, lp.lhs);

		ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);
	}
	ierr = PetscFinalize();
	return ierr;
}

