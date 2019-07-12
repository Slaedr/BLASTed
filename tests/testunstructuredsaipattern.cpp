/** \file
 * \brief Test SAI/ISAI pattern generation for a small unstructured grid
 */

#include <boost/align/align.hpp>
#include "srmatrixdefs.hpp"

using boost::alignemnt::aligned_alloc;

CRawBSRMatrix<double,int> generate_small_unstructured_matrix()
{
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
}

void test_sai_upper(const bool fullsai)
{
}

void test_sai_lower(const bool fullsai)
{
}

int main(int argv, char *argv[])
{
	test_sai(true);
	test_sai(false);

	return 0;
}
