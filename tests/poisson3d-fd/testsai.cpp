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

void test_fullmatrix_interior(const CartMesh& m, const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                              const LeftSAIPattern<int>& sp, const PetscInt testrow)
{
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
				//printf("  LHS row ind = %d. ", sp.browind[colstart+i]); fflush(stdout);
				assert(sp.browind[colstart+i] == i);
			}
			assert(sp.browind[colstart+6] == 12);
		}
		else if(jcol == start+1) {
			// down column
			assert(sp.browind[colstart] == 1);
			for(int i = 1; i < 5; i++)
				assert(sp.browind[colstart+i] == 5+i);
			assert(sp.browind[colstart+5] == 12);
			assert(sp.browind[colstart+6] == 19);
		}
		else if(jcol == start+2) {
			// left column
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 7);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 8+i);
			assert(sp.browind[colstart+5] == 15);
			assert(sp.browind[colstart+6] == 20);
		}
		else if(jcol == start+3) {
			// centre column
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 8);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 9+i);
			assert(sp.browind[colstart+5] == 16);
			assert(sp.browind[colstart+6] == 21);
		}
		else if(jcol == start+4) {
			// right column
			assert(sp.browind[colstart] == 4);
			assert(sp.browind[colstart+1] == 9);
			for(int i = 2; i < 5; i++)
				assert(sp.browind[colstart+i] == 10+i);
			assert(sp.browind[colstart+5] == 17);
			assert(sp.browind[colstart+6] == 22);
		}
		else if(jcol == start+5) {
			// up column
			assert(sp.browind[colstart] == 5);
			assert(sp.browind[colstart+1] == 12);
			for(int i = 2; i < 6; i++)
				assert(sp.browind[colstart+i] == 13+i);
			assert(sp.browind[colstart+6] == 23);
		}
		else if(jcol == start+6) {
			// front column
			assert(sp.browind[colstart] == 12);
			for(int i = 1; i < 7; i++)
				assert(sp.browind[colstart+i] == 18+i);
		}
	}
}

void test_fullmatrix_boundaryface(const CRawBSRMatrix<PetscScalar,PetscInt>& mat,
                                  const LeftSAIPattern<int>& sp, const PetscInt testrow)
{
}

int test_sai(const bool fullsai, const CartMesh& m, const Mat A)
{
	int ierr = 0;

	const CRawBSRMatrix<PetscScalar,PetscInt> mat = wrapLocalPetscMat(A, 1);

	const LeftSAIPattern<int> sp = left_SAI_pattern(mat);

	// Select some interior point
	const PetscInt testpoint[] = {3,3,3};
	assert(m.gnpoind(0) >= 5);
	assert(m.gnpoind(1) >= 7);
	assert(m.gnpoind(2) >= 6);

	const PetscInt testrow = testpoint[2]*(m.gnpoind(0)-2)*(m.gnpoind(1)-2)
		+ testpoint[1]*(m.gnpoind(0)-2) + testpoint[0];

	PetscInt ncols = 0;
	ierr = MatGetRow(A, testrow, &ncols, NULL, NULL); CHKERRQ(ierr);
	printf("Number of cols in row %d is %d.\n", testrow, ncols);
	assert(ncols == 7);

	// Test full matrix
	test_fullmatrix_interior(m, mat, sp, testrow);

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

		ierr = destroyDiscreteLinearProblem(&lp); CHKERRQ(ierr);
	}
	ierr = PetscFinalize();
	return ierr;
}

