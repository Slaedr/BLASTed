/** \file
 * \brief Test SAI/ISAI pattern generation for a small unstructured grid
 */

#undef NDEBUG
#include <stdexcept>
#include <vector>
#include <boost/align/align.hpp>
#include "srmatrixdefs.hpp"
#include "../src/sai.hpp"

using namespace blasted;

std::vector<std::vector<int>> generate_small_unstructured_adjlists()
{
	std::vector<std::vector<int>> al(13);

	al[0].resize(3);
	al[0][0] = 0; al[0][1] = 2; al[0][2] = 4;

	al[1].resize(3);
	al[1][0] = 1; al[1][1] = 2; al[1][2] = 5;

	al[2].resize(5);
	al[2][0] = 0; al[2][1] = 1; al[2][2] = 2; al[2][3] = 3; al[2][4] = 5;

	al[3].resize(5);
	al[3][0] = 2; al[3][1] = 3; al[3][2] = 6; al[3][3] = 9; al[3][4] = 12;

	al[4].resize(3);
	al[4][0] = 0; al[4][1] = 4; al[4][2] = 6;

	al[5].resize(5);
	al[5][0] = 1; al[5][1] = 2; al[5][2] = 5; al[5][3] = 7; al[5][4] = 8;

	al[6].resize(4);
	al[6][0] = 3; al[6][1] = 4; al[6][2] = 6; al[6][3] = 12;

	al[7].resize(3);
	al[7][0] = 5; al[7][1] = 7; al[7][2] = 8;

	al[8].resize(4);
	al[8][0] = 5; al[8][1] = 7; al[8][2] = 8; al[8][3] = 9;

	al[9].resize(4);
	al[9][0] = 3; al[9][1] = 8; al[9][2] = 9; al[9][3] = 10;

	al[10].resize(3); al[10][0] = 9; al[10][1] = 10; al[10][2] = 11;

	al[11].resize(3); al[11][0] = 10; al[11][1] = 11; al[11][2] = 12;

	al[12].resize(4); al[12][0] = 3; al[12][1] = 6; al[12][2] = 11; al[12][3] = 12;

	return al;
}

int getCSRPosition(const std::vector<std::vector<int>>& meshadj, const int cell, const int nbridx)
{
	int pos = 0;
	for(int i = 0; i < cell; i++)
		pos += meshadj[i].size();

	pos += nbridx;
	return pos;
}

int getUpperCSRPosition(const std::vector<std::vector<int>>& meshadj, const int cell, const int nbridx)
{
	if(meshadj[cell][nbridx] < cell)
		throw std::runtime_error("Requested neighbour is not upper!");

	int pos = 0;
	for(int i = 0; i < cell; i++)
		for(int j = 0; j < static_cast<int>(meshadj[i].size()); j++)
			if(meshadj[i][j] >= i)
				pos++;

	int selfidx = -1;
	for(int j = 0; j < static_cast<int>(meshadj[cell].size()); j++)
		if(meshadj[cell][j] == cell)
			selfidx = j;
	if(selfidx == -1)
		throw std::runtime_error("Problem with adj lists!");

	pos += nbridx-selfidx;
	return pos;
}

// Mesh corresponding to generate_small_unstructured_adjlists
SRMatrixStorage<double,int> generate_small_unstructured_matrix()
{
	SRMatrixStorage<double,int> mat;
	mat.nbrows = 13;

	mat.browptr.resize(14);

	mat.browptr[0] = 0;
	mat.browptr[1] = 3; mat.browptr[2] = 3; mat.browptr[3] = 5; mat.browptr[4] = 5; mat.browptr[5] = 3;
	mat.browptr[6] = 5;
	mat.browptr[7] = 4; mat.browptr[8] = 3; mat.browptr[9] = 4; mat.browptr[10] = 4;
	mat.browptr[11] = 3; mat.browptr[12] = 3;
	mat.browptr[13] = 4;

	for(int i = 0; i < mat.nbrows; i++)
		mat.browptr[i+1] += mat.browptr[i];
	assert(mat.browptr[mat.nbrows] == 49);
	mat.nnzb = mat.browptr[mat.nbrows];
	mat.nbstored = mat.nnzb;

	mat.bcolind.resize(mat.nnzb);
	mat.vals.resize(mat.nnzb);
	mat.diagind.resize(mat.nbrows);

	mat.bcolind[mat.browptr[0]] = 0; mat.bcolind[mat.browptr[0]+1] = 2; mat.bcolind[mat.browptr[0]+2] = 4;
	mat.diagind[0] = mat.browptr[0]; assert(mat.browptr[1] == mat.browptr[0]+3);

	mat.bcolind[mat.browptr[1]] = 1; mat.bcolind[mat.browptr[1]+1] = 2; mat.bcolind[mat.browptr[1]+2] = 5;
	mat.diagind[1] = mat.browptr[1]; assert(mat.browptr[2] == mat.browptr[1]+3);

	mat.bcolind[mat.browptr[2]] = 0; mat.bcolind[mat.browptr[2]+1] = 1; mat.bcolind[mat.browptr[2]+2] = 2;
	mat.bcolind[mat.browptr[2]+3] = 3;
	mat.bcolind[mat.browptr[2]+4] = 5;
	mat.diagind[2] = mat.browptr[2]+2; assert(mat.browptr[3] == mat.browptr[2]+5);

	mat.bcolind[mat.browptr[3]] = 2; mat.bcolind[mat.browptr[3]+1] = 3; mat.bcolind[mat.browptr[3]+2] = 6;
	mat.bcolind[mat.browptr[3]+3] = 9;
	mat.bcolind[mat.browptr[3]+4] = 12;
	mat.diagind[3] = mat.browptr[3]+1; assert(mat.browptr[4] == mat.browptr[3]+5);

	mat.bcolind[mat.browptr[4]] = 0; mat.bcolind[mat.browptr[4]+1] = 4; mat.bcolind[mat.browptr[4]+2] = 6;
	mat.diagind[4] = mat.browptr[4]+1; assert(mat.browptr[5] == mat.browptr[4]+3);

	mat.bcolind[mat.browptr[5]] = 1; mat.bcolind[mat.browptr[5]+1] = 2; mat.bcolind[mat.browptr[5]+2] = 5;
	mat.bcolind[mat.browptr[5]+3] = 7;
	mat.bcolind[mat.browptr[5]+4] = 8;
	mat.diagind[5] = mat.browptr[5]+2; assert(mat.browptr[6] == mat.browptr[5]+5);

	mat.bcolind[mat.browptr[6]] = 3; mat.bcolind[mat.browptr[6]+1] = 4; mat.bcolind[mat.browptr[6]+2] = 6;
	mat.bcolind[mat.browptr[6]+3] = 12;
	mat.diagind[6] = mat.browptr[6]+2; assert(mat.browptr[7] == mat.browptr[6]+4);

	mat.bcolind[mat.browptr[7]] = 5; mat.bcolind[mat.browptr[7]+1] = 7; mat.bcolind[mat.browptr[7]+2] = 8;
	mat.diagind[7] = mat.browptr[7]+1; assert(mat.browptr[8] == mat.browptr[7]+3);

	mat.bcolind[mat.browptr[8]] = 5; mat.bcolind[mat.browptr[8]+1] = 7; mat.bcolind[mat.browptr[8]+2] = 8;
	mat.bcolind[mat.browptr[8]+3] = 9;
	mat.diagind[8] = mat.browptr[8]+2; assert(mat.browptr[9] == mat.browptr[8]+4);

	mat.bcolind[mat.browptr[9]] = 3; mat.bcolind[mat.browptr[9]+1] = 8; mat.bcolind[mat.browptr[9]+2] = 9;
	mat.bcolind[mat.browptr[9]+3] = 10;
	mat.diagind[9] = mat.browptr[9]+2; assert(mat.browptr[10] == mat.browptr[9]+4);

	mat.bcolind[mat.browptr[10]] = 9; mat.bcolind[mat.browptr[10]+1] = 10; mat.bcolind[mat.browptr[10]+2] = 11;
	mat.diagind[10] = mat.browptr[10]+1; assert(mat.browptr[11] == mat.browptr[10]+3);

	mat.bcolind[mat.browptr[11]] = 10; mat.bcolind[mat.browptr[11]+1] = 11; mat.bcolind[mat.browptr[11]+2] = 12;
	mat.diagind[11] = mat.browptr[11]+1; assert(mat.browptr[12] == mat.browptr[11]+3);

	mat.bcolind[mat.browptr[12]] = 3; mat.bcolind[mat.browptr[12]+1] = 6; mat.bcolind[mat.browptr[12]+2] = 11;
	mat.bcolind[mat.browptr[12]+3] = 12;
	mat.diagind[12] = mat.browptr[12]+3; assert(mat.browptr[13] == mat.browptr[12]+4);

	for(int i = 0; i < mat.nnzb; i++)
		mat.vals[i] = -0.1;
	for(int i = 0; i < mat.nbrows; i++)
		mat.vals[mat.diagind[i]] = 4.0;

	mat.browendptr.wrap(&mat.browptr[1], mat.nbrows);

	return mat;
}

void test_sai(const bool fullsai)
{
	const SRMatrixStorage<double,int> tmat = generate_small_unstructured_matrix();

	if(fullsai)
	{
		const LeftSAIPattern<int> sp = left_SAI_pattern(share_with_const<double,int>(tmat));

		const int testcell = 3;

		assert(sp.nVars[testcell] == 5);
		assert(sp.nEqns[testcell] == 12);
		assert(sp.localCentralRow[testcell] == 3);

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
		const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(share_with_const(tmat));

		const int testcell = 3;

		assert(sp.nVars[testcell] == 5);
		assert(sp.nEqns[testcell] == 5);
		assert(sp.localCentralRow[testcell] == 1);

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
				assert(sp.bpos[spk+1] == tmat.browptr[colind]+2);
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

void test_sai_boundary(const bool fullsai)
{
	const SRMatrixStorage<double,int> tmat = generate_small_unstructured_matrix();

	if(fullsai)
	{
		const LeftSAIPattern<int> sp = left_SAI_pattern(share_with_const(tmat));

		const int testcell = 6;

		assert(sp.nVars[testcell] == 4);
		assert(sp.nEqns[testcell] == 8);

		const int start = sp.sairowptr[testcell], end = sp.sairowptr[testcell+1];
		assert(end-start == 4);
		assert(sp.bcolptr[end]-sp.bcolptr[start] == 16);

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
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 4);
			assert(sp.browind[colstart+3] == 5);
			assert(sp.browind[colstart+4] == 7);
		}
		{
			const int colstart = sp.bcolptr[start+1];
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.browind[colstart+2] == 4);
		}
		{
			const int colstart = sp.bcolptr[start+2];
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 3);
			assert(sp.browind[colstart+2] == 4);
			assert(sp.browind[colstart+3] == 7);
		}
		{
			const int colstart = sp.bcolptr[start+3];
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 4);
			assert(sp.browind[colstart+2] == 6);
			assert(sp.browind[colstart+3] == 7);
		}

		printf(" >> Test for SAI at boundary point passed.\n"); fflush(stdout);
	}
	else
	{
		const LeftSAIPattern<int> sp = left_incomplete_SAI_pattern(share_with_const(tmat));

		const int testcell = 6;

		assert(sp.nVars[testcell] == 4);
		assert(sp.nEqns[testcell] == 4);

		const int start = sp.sairowptr[testcell], end = sp.sairowptr[testcell+1];
		assert(end-start == 4);
		assert(sp.bcolptr[end]-sp.bcolptr[start] == 12);

		// number of non-zeros in each column of the ISAI LHS for the test cell
		assert(sp.bcolptr[start+1] - sp.bcolptr[start] == 3);
		assert(sp.bcolptr[start+2] - sp.bcolptr[start+1] == 2);
		assert(sp.bcolptr[start+3] - sp.bcolptr[start+2] == 4);
		assert(sp.bcolptr[start+4] - sp.bcolptr[start+3] == 3);

		// check positions
		for(int jj = tmat.browptr[testcell], spj = start; jj < tmat.browendptr[testcell]; jj++, spj++)
		{
			assert(spj < end);

			const int colind = tmat.bcolind[jj];
			const int spk = sp.bcolptr[spj];
			if(colind == 3) {
				assert(sp.bpos[spk] == tmat.diagind[colind]);
				assert(sp.bpos[spk+1] == tmat.diagind[colind]+1);
				assert(sp.bpos[spk+2] == tmat.diagind[colind]+3);
			}
			else if(colind == 4) {
				assert(sp.bpos[spk] == tmat.diagind[colind]);
				assert(sp.bpos[spk+1] == tmat.diagind[colind]+1);
			}
			else if(colind == 6) {
				for(int j = 0; j < 4; j++)
					assert(sp.bpos[spk+j] == tmat.browptr[colind]+j);
			}
			else if(colind == 12) {
				assert(sp.bpos[spk] == tmat.browptr[colind]);
				assert(sp.bpos[spk+1] == tmat.browptr[colind]+1);
				assert(sp.bpos[spk+2] == tmat.diagind[colind]);
			}
			else {
				throw std::runtime_error("Bad column index!");
			}
		}

		// Check local row indices
		{
			const int colstart = sp.bcolptr[start];
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 3);
		}
		{
			const int colstart = sp.bcolptr[start+1];
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 2);
		}
		{
			const int colstart = sp.bcolptr[start+2];
			for(int j = 0; j < 4; j++)
				assert(sp.browind[colstart+j] == j);
		}
		{
			const int colstart = sp.bcolptr[start+3];
			assert(sp.browind[colstart] == 0);
			assert(sp.browind[colstart+1] == 2);
			assert(sp.browind[colstart+2] == 3);
		}

		printf(" >> Test for incomplete SAI at boundary point passed.\n"); fflush(stdout);
	}
}

void test_sai_upper(const bool fullsai)
{
	const SRMatrixStorage<const double,const int> mat = move_to_const(generate_small_unstructured_matrix());
	const SRMatrixStorage<const double,const int> tmat
		= getUpperTriangularView(std::forward<const SRMatrixStorage<const double,const int>>(mat));

	if(fullsai)
	{
		const LeftSAIPattern<int> sp = left_SAI_pattern(std::move(tmat));

		const int testcell = 3;

		assert(sp.nVars[testcell] == 4);
		assert(sp.nEqns[testcell] == 5);

		const int start = sp.sairowptr[testcell], end = sp.sairowptr[testcell+1];
		assert(end-start == 4);
		assert(sp.bcolptr[end]-sp.bcolptr[start] == 9);

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
			for(int j = 0; j < 3; j++)
				assert(sp.browind[colstart+j] == j);
			assert(sp.browind[colstart+3] == 4);
		}
		{
			const int colstart = sp.bcolptr[start+1];
			assert(sp.browind[colstart] == 1);
			assert(sp.browind[colstart+1] == 4);
		}
		{
			const int colstart = sp.bcolptr[start+2];
			assert(sp.browind[colstart] == 2);
			assert(sp.browind[colstart+1] == 3);
		}
		{
			const int colstart = sp.bcolptr[start+3];
			assert(sp.browind[colstart] == 4);
		}

		printf(" >> Test for SAI if upper triangular matrix at interior point passed.\n"); fflush(stdout);
	}
}

int main(int argc, char *argv[])
{
	test_sai(true);
	test_sai(false);
	test_sai_upper(true);
	test_sai_boundary(true);
	test_sai_boundary(false);

	return 0;
}

