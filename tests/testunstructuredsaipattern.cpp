/** \file
 * \brief Test SAI/ISAI pattern generation for a small unstructured grid
 */

#include <stdexcept>
#include <boost/align/align.hpp>
#include "srmatrixdefs.hpp"
#include "../src/sai.hpp"

using namespace blasted;

CRawBSRMatrix<double,int> generate_small_unstructured_matrix()
{
	using boost::alignment::aligned_alloc;

	CRawBSRMatrix<double,int> mat;
	mat.nbrows = 13;

	int *rowptr = (int*)aligned_alloc(CACHE_LINE_LEN, 14*sizeof(int));
	rowptr[0] = 0;
	rowptr[1] = 3; rowptr[2] = 3; rowptr[3] = 5; rowptr[4] = 5; rowptr[5] = 3; rowptr[6] = 5;
	rowptr[7] = 4; rowptr[8] = 3; rowptr[9] = 4; rowptr[10] = 4; rowptr[11] = 3; rowptr[12] = 3;
	rowptr[13] = 4;

	for(int i = 0; i < mat.nbrows; i++)
		rowptr[i+1] += rowptr[i];
	assert(rowptr[mat.nbrows] == 49);
	mat.nnzb = rowptr[mat.nbrows];
	mat.nbstored = mat.nnzb;

	int *colind = (int*)aligned_alloc(CACHE_LINE_LEN, mat.nnzb*sizeof(int));
	double *vals = (double*)aligned_alloc(CACHE_LINE_LEN, mat.nnzb*sizeof(double));
	int *diagind = (int*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(int));

	colind[rowptr[0]] = 0; colind[rowptr[0]+1] = 2; colind[rowptr[0]+2] = 4;
	diagind[0] = rowptr[0]; assert(rowptr[1] == rowptr[0]+3);

	colind[rowptr[1]] = 1; colind[rowptr[1]+1] = 2; colind[rowptr[1]+2] = 5;
	diagind[1] = rowptr[1]; assert(rowptr[2] == rowptr[1]+3);

	colind[rowptr[2]] = 0; colind[rowptr[2]+1] = 1; colind[rowptr[2]+2] = 2; colind[rowptr[2]+3] = 3;
	colind[rowptr[2]+4] = 5;
	diagind[2] = rowptr[2]+2; assert(rowptr[3] == rowptr[2]+5);

	colind[rowptr[3]] = 2; colind[rowptr[3]+1] = 3; colind[rowptr[3]+2] = 6; colind[rowptr[3]+3] = 9;
	colind[rowptr[3]+4] = 12;
	diagind[3] = rowptr[3]+1; assert(rowptr[4] == rowptr[3]+5);

	colind[rowptr[4]] = 0; colind[rowptr[4]+1] = 4; colind[rowptr[4]+2] = 6;
	diagind[4] = rowptr[4]+1; assert(rowptr[5] == rowptr[4]+3);

	colind[rowptr[5]] = 1; colind[rowptr[5]+1] = 2; colind[rowptr[5]+2] = 5; colind[rowptr[5]+3] = 7;
	colind[rowptr[5]+4] = 8;
	diagind[5] = rowptr[5]+2; assert(rowptr[6] == rowptr[5]+5);

	colind[rowptr[6]] = 3; colind[rowptr[6]+1] = 4; colind[rowptr[6]+2] = 6; colind[rowptr[6]+3] = 12;
	diagind[6] = rowptr[6]+2; assert(rowptr[7] == rowptr[6]+4);

	colind[rowptr[7]] = 5; colind[rowptr[7]+1] = 7; colind[rowptr[7]+2] = 8;
	diagind[7] = rowptr[7]+1; assert(rowptr[8] == rowptr[7]+3);

	colind[rowptr[8]] = 5; colind[rowptr[8]+1] = 7; colind[rowptr[8]+2] = 8; colind[rowptr[8]+3] = 9;
	diagind[8] = rowptr[8]+2; assert(rowptr[9] == rowptr[8]+4);

	colind[rowptr[9]] = 3; colind[rowptr[9]+1] = 8; colind[rowptr[9]+2] = 9; colind[rowptr[9]+3] = 10;
	diagind[9] = rowptr[9]+2; assert(rowptr[10] == rowptr[9]+4);

	colind[rowptr[10]] = 9; colind[rowptr[10]+1] = 10; colind[rowptr[10]+2] = 11;
	diagind[10] = rowptr[10]+1; assert(rowptr[11] == rowptr[10]+3);

	colind[rowptr[11]] = 10; colind[rowptr[11]+1] = 11; colind[rowptr[11]+2] = 12;
	diagind[11] = rowptr[11]+1; assert(rowptr[12] == rowptr[11]+3);

	colind[rowptr[12]] = 3; colind[rowptr[12]+1] = 6; colind[rowptr[12]+2] = 11; colind[rowptr[12]+3] = 12;
	diagind[12] = rowptr[12]+3; assert(rowptr[13] == rowptr[12]+4);

	for(int i = 0; i < mat.nnzb; i++)
		vals[i] = -0.1;
	for(int i = 0; i < mat.nbrows; i++)
		vals[diagind[i]] = 4.0;

	mat.browptr = rowptr;
	mat.bcolind = colind;
	mat.diagind = diagind;
	mat.browendptr = rowptr+1;
	mat.vals = vals;

	return mat;
}

void test_sai(const bool fullsai)
{
	const CRawBSRMatrix<double,int> tmat = generate_small_unstructured_matrix();
	if(fullsai)
	{
		const LeftSAIPattern<int> sp = left_SAI_pattern(tmat);

		const int testcell = 3;

		assert(sp.nVars[testcell] == 5);
		assert(sp.nEqns[testcell] == 12);

		const int start = sp.sairowptr[testcell], end = sp.sairowptr[testcell+1];
		assert(end-start == 5);
		assert(sp.bcolptr[end]-sp.bcolptr[start] == 22);

		// check positions
		for(int jj = tmat.browptr[testcell], spj = start; jj < tmat.browendptr[testcell]; jj++, spj++)
		{
			assert(spj < end);

			const int colind = tmat.bcolind[jj];
			assert(sp.bcolptr[spj+1] - sp.bcolptr[spj] == tmat.browendptr[colind] - tmat.browptr[colind]);

			for(int kk = tmat.browptr[colind], spk = sp.bcolptr[spj];
			    kk < tmat.browendptr[colind]; kk++, spk++)
			{
				assert(spk < sp.bcolptr[spj+1]);
				assert(kk == sp.bpos[spk]);
			}
		}

		// Check local row indices
		{
			const int colstart = sp.bcolptr[start];
			for(int j = 0; j < 4; j++)
				assert(sp.browind[colstart+j] == j);
			assert(sp.browind[colstart+4] == 5);
		}
		{
			const int colstart = sp.bcolptr[start+1];
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.browind[colstart+2] == 6);
			assert(sp.browind[colstart+3] == 8);
			assert(sp.browind[colstart+4] == 11);
		}
		{
			const int colstart = sp.bcolptr[start+2];
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.browind[colstart+2] == 6);
			assert(sp.browind[colstart+3] == 11);
		}
		{
			const int colstart = sp.bcolptr[start+3];
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 7);
			assert(sp.browind[colstart+2] == 8);
			assert(sp.browind[colstart+3] == 9);
		}
		{
			const int colstart = sp.bcolptr[start+4];
			assert(sp.browind[colstart] == 3);
			assert(sp.browind[colstart+1] == 6);
			assert(sp.browind[colstart+2] == 10);
			assert(sp.browind[colstart+3] == 11);
		}

		printf(" >> Test for SAI at interior point passed.\n"); fflush(stdout);
	}
	else {
		const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(tmat);

		const int testcell = 3;

		assert(sp.nVars[testcell] == 5);
		assert(sp.nEqns[testcell] == 5);

		const int start = sp.sairowptr[testcell], end = sp.sairowptr[testcell+1];
		assert(end-start == 5);
		assert(sp.bcolptr[end]-sp.bcolptr[start] == 15);

		// number of non-zeros in each column of the ISAI LHS for the test cell
		assert(sp.bcolptr[start+1] - sp.bcolptr[start] == 2);
		assert(sp.bcolptr[start+2] - sp.bcolptr[start+1] == 5);
		assert(sp.bcolptr[start+3] - sp.bcolptr[start+2] == 3);
		assert(sp.bcolptr[start+4] - sp.bcolptr[start+3] == 2);
		assert(sp.bcolptr[start+5] - sp.bcolptr[start+4] == 3);

		// check positions
		for(int jj = tmat.browptr[testcell], spj = start; jj < tmat.browendptr[testcell]; jj++, spj++)
		{
			assert(spj < end);

			const int colind = tmat.bcolind[jj];
			const int spk = sp.bcolptr[spj];
			if(colind == 2) {
				assert(sp.bpos[spk] == tmat.diagind[colind]);
				assert(sp.bpos[spk+1] == tmat.diagind[colind]+1);
			}
			else if(colind == 3) {
				for(int j = 0; j < 5; j++)
					assert(sp.bpos[spk+j] == tmat.browptr[colind]+j);
			}
			else if(colind == 6) {
				assert(sp.bpos[spk] == tmat.browptr[colind]);
				assert(sp.bpos[spk+1] == tmat.diagind[colind]);
				/*extra*/ assert(sp.bpos[spk+1] == tmat.browptr[colind]+2);
				assert(sp.bpos[spk+2] == tmat.browptr[colind]+3);
			}
			else if(colind == 9) {
				assert(sp.bpos[spk] == tmat.browptr[colind]);
				assert(sp.bpos[spk+1] == tmat.diagind[colind]);
			}
			else if(colind == 12) {
				assert(sp.bpos[spk] == tmat.browptr[colind]);
				assert(sp.bpos[spk+1] == tmat.browptr[colind]+1);
				assert(sp.bpos[spk+2] == tmat.diagind[colind]);
				assert(sp.bpos[spk+2] == tmat.browptr[colind]+3);
			}
			else {
				throw std::runtime_error("Bad column index!");
			}
		}

		// Check local row indices
		{
			const int colstart = sp.bcolptr[start];
			for(int j = 0; j < 2; j++)
				assert(sp.browind[colstart+j] == j);
		}
		{
			const int colstart = sp.bcolptr[start+1];
			for(int j = 0; j < 5; j++)
				assert(sp.browind[colstart+j] == j);
		}
		{
			const int colstart = sp.bcolptr[start+2];
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 4);
		}
		{
			const int colstart = sp.bcolptr[start+3];
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 3);
		}
		{
			const int colstart = sp.bcolptr[start+4];
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 4);
		}

		printf(" >> Test for incomplete SAI at interior point passed.\n"); fflush(stdout);
	}
}

void test_sai_upper(const bool fullsai)
{
}

void test_sai_lower(const bool fullsai)
{
}

int main(int argc, char *argv[])
{
	test_sai(true);
	test_sai(false);

	return 0;
}
