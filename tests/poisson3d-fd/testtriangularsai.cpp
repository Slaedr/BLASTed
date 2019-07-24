#undef NDEBUG
#include <cassert>

#include "srmatrixdefs.hpp"
#include "../../src/sai.hpp"
#include "../testutils.hpp"
#include "poisson_setup.h"
#include "poisson_utils.hpp"
#include "testsai.hpp"

using namespace blasted;

void test_fullsai_uppertri_interior(const CartMesh& m,
                                    const SRMatrixStorage<const PetscScalar,const PetscInt>& mat,
                                    const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
	assert(testpoint[0] >= 2);
	assert(testpoint[1] >= 2);
	assert(testpoint[2] >= 2);
	assert(testpoint[0] <= m.gnpoind(0)-4);
	assert(testpoint[1] <= m.gnpoind(1)-4);
	assert(testpoint[2] <= m.gnpoind(2)-4);

	const PetscInt rightpoint[] = { testpoint[0]+1, testpoint[1], testpoint[2] };
	const PetscInt rightrow = getMatRowIdx(m, rightpoint);
	const PetscInt uppoint[] = { testpoint[0], testpoint[1]+1, testpoint[2] };
	const PetscInt uprow = getMatRowIdx(m, uppoint);
	const PetscInt frontpoint[] = { testpoint[0], testpoint[1], testpoint[2]+1 };
	const PetscInt frontrow = getMatRowIdx(m, frontpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	assert(sp.nVars[testrow] == 4);
	assert(sp.nEqns[testrow] == 10);
	assert(sp.localCentralRow[testrow] == 0);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == sp.nVars[testrow]);

	for(int jcol = firstcol; jcol < lastcol; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
		assert(colsize == 4);

		if(jcol == firstcol) {
			// Back
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 1);
			assert(sp.browind[colstart+2] == 3);
			assert(sp.browind[colstart+3] == 6);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.diagind[testrow]+j);
		}
		else if(jcol == firstcol+1) {
			// right
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 4);
			assert(sp.browind[colstart+3] == 7);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.diagind[rightrow]+j);
		}
		else if(jcol == firstcol+2) {
			// right
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.browind[colstart+2] == 5);
			assert(sp.browind[colstart+3] == 8);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.diagind[uprow]+j);
		}
		else if(jcol == firstcol+3) {
			// right
			assert(sp.browind[colstart] == 6);
			assert(sp.browind[colstart+1] == 7);
			assert(sp.browind[colstart+2] == 8);
			assert(sp.browind[colstart+3] == 9);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.diagind[frontrow]+j);
		}
		else {
			throw "Invalid column!";
		}
	}

	printf(" >> Full SAI upper triangular test for interior point passed.\n");
}

/// Test SAI pattern on lower triangular matrix at an interior point of a structured 3D grid
void test_fullsai_lowertri_interior(const CartMesh& m,
                                    const SRMatrixStorage<const PetscScalar,const PetscInt>& mat,
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
	assert(sp.nEqns[testrow] == 10);
	assert(sp.localCentralRow[testrow] == 9);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == sp.nVars[testrow]);

	for(int jcol = firstcol; jcol < lastcol; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
		assert(colsize == 4);

		if(jcol == firstcol) {
			// Back
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 1);
			assert(sp.browind[colstart+2] == 2);
			assert(sp.browind[colstart+3] == 3);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[backrow]+j);
		}
		else if(jcol == firstcol+1) {
			// down
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.browind[colstart+2] == 5);
			assert(sp.browind[colstart+3] == 6);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[downrow]+j);
		}
		else if(jcol == firstcol+2) {
			// left
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 5);
			assert(sp.browind[colstart+2] == 7);
			assert(sp.browind[colstart+3] == 8);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[leftrow]+j);
		}
		else if(jcol == firstcol+3) {
			// centre
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 6);
			assert(sp.browind[colstart+2] == 8);
			assert(sp.browind[colstart+3] == 9);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[testrow]+j);
		}
		else {
			throw "Invalid column!";
		}
	}

	printf(" >> Full SAI lower triangular test for interior point passed.\n");
}

/// Test the SAI or incomplete SAI pattern of an upper triangular matrix for a point on the i+ boundary
void test_uppertri_boundary_i_end(const bool fullsai, const CartMesh& m,
                                  const SRMatrixStorage<const PetscScalar,const PetscInt>& mat,
                                  const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
	assert(testpoint[0] == m.gnpoind(0)-2);
	assert(testpoint[1] >= 3 && testpoint[1] <= m.gnpoind(1)-4);
	assert(testpoint[2] >= 3 && testpoint[2] <= m.gnpoind(2)-4);

	const PetscInt uppoint[] = { testpoint[0], testpoint[1]+1, testpoint[2] };
	const PetscInt uprow = getMatRowIdx(m, uppoint);
	const PetscInt frontpoint[] = { testpoint[0], testpoint[1], testpoint[2]+1 };
	const PetscInt frontrow = getMatRowIdx(m, frontpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == sp.nVars[testrow]);

	if(fullsai) {
		assert(sp.nVars[testrow] == 3);
		assert(sp.nEqns[testrow] == 6);
		assert(sp.localCentralRow[testrow] == 0);

		for(int jcol = firstcol; jcol < lastcol; jcol++)
		{
			const int colstart = sp.bcolptr[jcol];

			const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
			assert(colsize == 3);

			if(jcol == firstcol) {
				// Centre
				assert(sp.browind[colstart] == 0);
				assert(sp.browind[colstart+1] == 1);
				assert(sp.browind[colstart+2] == 3);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[testrow]+j);
			}
			else if(jcol == firstcol+1) {
				// up
				assert(sp.browind[colstart] == 1);
				assert(sp.browind[colstart+1] == 2);
				assert(sp.browind[colstart+2] == 4);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[uprow]+j);
			}
			else if(jcol == firstcol+2) {
				// front
				assert(sp.browind[colstart] == 3);
				assert(sp.browind[colstart+1] == 4);
				assert(sp.browind[colstart+2] == 5);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[frontrow]+j);
			}
			else {
				throw "Invalid column!";
			}
		}

		printf(" >> Full SAI upper triangular test for boundary point at i-end passed.\n");
	}
	else {
		assert(sp.nVars[testrow] == 3);
		assert(sp.nEqns[testrow] == 3);
		assert(sp.localCentralRow[testrow] == 0);

		for(int jcol = firstcol; jcol < lastcol; jcol++)
		{
			const int colstart = sp.bcolptr[jcol];

			const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
			if(jcol == firstcol)
				assert(colsize == 3);
			else
				assert(colsize == 1);

			if(jcol == firstcol) {
				// Centre
				assert(sp.browind[colstart] == 0);
				assert(sp.browind[colstart+1] == 1);
				assert(sp.browind[colstart+2] == 2);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[testrow]+j);
			}
			else if(jcol == firstcol+1) {
				// up
				assert(sp.browind[colstart] == 1);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[uprow]+j);
			}
			else if(jcol == firstcol+2) {
				// front
				assert(sp.browind[colstart] == 2);

				for(int j = 0; j < colsize; j++)
					assert(sp.bpos[colstart+j] == mat.diagind[frontrow]+j);
			}
			else {
				throw "Invalid column!";
			}
		}

		printf(" >> Incomplete SAI upper triangular test for boundary point at i-end passed.\n");
	}
}

/// Test the SAI pattern for a point on the j- boundary
void test_fullsai_lowertri_boundary_j_start(const CartMesh& m,
                                            const SRMatrixStorage<const PetscScalar,const PetscInt>& mat,
                                            const LeftSAIPattern<int>& sp, const PetscInt testpoint[3])
{
	assert(testpoint[0] >= 3 && testpoint[0] <= m.gnpoind(0)-4);
	assert(testpoint[1] == 1);
	assert(testpoint[2] >= 3 && testpoint[0] <= m.gnpoind(2)-4);

	const PetscInt backpoint[] = { testpoint[0], testpoint[1], testpoint[2]-1 };
	const PetscInt backrow = getMatRowIdx(m, backpoint);
	const PetscInt leftpoint[] = { testpoint[0]-1, testpoint[1], testpoint[2] };
	const PetscInt leftrow = getMatRowIdx(m, leftpoint);

	const PetscInt testrow = getMatRowIdx(m, testpoint);

	assert(sp.nVars[testrow] == 3);
	assert(sp.nEqns[testrow] == 6);
	assert(sp.localCentralRow[testrow] == 5);

	const int firstcol = sp.sairowptr[testrow], lastcol = sp.sairowptr[testrow+1];
	assert(lastcol-firstcol == sp.nVars[testrow]);

	for(int jcol = firstcol; jcol < lastcol; jcol++)
	{
		const int colstart = sp.bcolptr[jcol];

		const int colsize = sp.bcolptr[jcol+1]-sp.bcolptr[jcol];
		assert(colsize == 3);

		if(jcol == firstcol) {
			// Back
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 1);
			assert(sp.browind[colstart+2] == 2);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[backrow]+j);
		}
		else if(jcol == firstcol+1) {
			// left
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.browind[colstart+2] == 4);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[leftrow]+j);
		}
		else if(jcol == firstcol+2) {
			// centre
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.browind[colstart+2] == 5);

			for(int j = 0; j < colsize; j++)
				assert(sp.bpos[colstart+j] == mat.browptr[testrow]+j);
		}
		else {
			throw "Invalid column!";
		}
	}

	printf(" >> Full SAI lower triangular test for boundary point at j-start passed.\n");
}

void test_incomplete_lowertri_interior(const CartMesh& m,
                                       const SRMatrixStorage<const PetscScalar,const PetscInt>& mat,
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
	assert(sp.localCentralRow[testrow] == 3);

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

	printf(" >> Incomplete SAI lower triangular test for interior point passed.\n");
}

int test_sai(const bool fullsai, const bool upper, const CartMesh& m, const Mat A)
{
	int ierr = 0;

	const SRMatrixStorage<const PetscScalar,const PetscInt> mat = wrapLocalPetscMat(A, 1);

	const SRMatrixStorage<const PetscScalar,const PetscInt> tmat = upper ?
		getUpperTriangularView(std::move(mat)) : getLowerTriangularView(std::move(mat));

	// Sanity check of triangular view
	assert(mat.nbrows == tmat.nbrows);
	for(int irow = 0; irow < tmat.nbrows; irow++)
		assert(tmat.diagind[irow] == mat.diagind[irow]);

	// Test interior point
	{
		const PetscInt testpoint[] = {3,3,3};
		assert(m.gnpoind(0) >= 5);
		assert(m.gnpoind(1) >= 7);
		assert(m.gnpoind(2) >= 6);

		const PetscInt testrow = getMatRowIdx(m, testpoint);

		PetscInt ncols = 0;
		ierr = MatGetRow(A, testrow, &ncols, NULL, NULL); CHKERRQ(ierr);
		printf(" Number of cols in row %d is %d.\n", testrow, ncols);

		if(fullsai) {
			const LeftSAIPattern<int> sp = left_SAI_pattern(std::move(tmat));
			if(upper)
				test_fullsai_uppertri_interior(m, tmat, sp, testpoint);
			else {
				test_fullsai_lowertri_interior(m, tmat, sp, testpoint);
			}
		}
		else {
			const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(std::move(tmat));
			if(upper) {
				// nothing yet
			}
			else
				test_incomplete_lowertri_interior(m, tmat, sp, testpoint);
		}
	}

	// Test +i boundary point for upper triangular matrix
	if(upper)
	{
		const PetscInt testpoint[] = {m.gnpoind(0)-2, 3, 3};
		const LeftSAIPattern<int> sp = fullsai ? left_SAI_pattern(std::move(tmat))
			: left_incomplete_SAI_pattern(std::move(tmat));

		test_uppertri_boundary_i_end(fullsai, m, tmat, sp, testpoint);
	}
	// Test -j boundary point for lower triangular matrix
	else
	{
		const PetscInt testpoint[] = {3,1,3};
		if(fullsai) {
			const LeftSAIPattern<int> sp = left_SAI_pattern(std::move(tmat));
			test_fullsai_lowertri_boundary_j_start(m, tmat, sp, testpoint);
		}
		else {
			// nothing yet
		}
	}

	return ierr;
}

int main(int argc, char *argv[])
{
	assert(argc > 3);

	const std::string confile = argv[1];
	const std::string test_type = argv[2];
	const std::string uplow = argv[3];

	int ierr = PetscInitialize(&argc, &argv, NULL, NULL);
	//petsc_throw(ierr);

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
