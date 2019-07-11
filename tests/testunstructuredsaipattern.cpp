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
	colind[rowptr[1]] = 1;

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
