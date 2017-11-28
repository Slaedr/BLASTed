/** \file testcsrmatrix.cpp
 * \brief Implementation of tests for CSR matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include "testcsrmatrix.hpp"

template <typename scalar>
TestCSRMatrix<scalar>::TestCSRMatrix(const int nbuildsweeps, const int napplysweeps)
	: BSRMatrix<scalar,int,1>(nbuildsweeps, napplysweeps)
{ }

/** The file containing the data to check against has
 * line 1: nrows ncols nnz
 * line 2: row-pointers
 * line 3: row-indices
 * line 4: column-indices
 * line 5: non-zero values
 * line 6: diagonal pointers
 */
template <typename scalar>
int TestCSRMatrix<scalar>::testStorage(const std::string compare_file)
{
	// Read file and compare with internal storage
	std::ifstream sortedfin(compare_file);
	if(!sortedfin) {
		std::cout << " CSRMatrix: File for comparison could not be opened to read!\n";
		std::abort();
	}

	int snnz, snrows, sncols;
	sortedfin >> snrows >> sncols >> snnz;

	assert(snnz == mat.browptr[mat.nbrows]);
	assert(snrows == mat.nbrows);

	std::vector<int> srowinds(snnz);
	std::vector<int> srowptr(snrows+1);
	std::vector<int> scolinds(snnz);
	std::vector<double> svals(snnz);
	std::vector<int> sdinds(snrows);

	for(int i = 0; i < snrows+1; i++)
		sortedfin >> srowptr[i];

	assert(srowptr[snrows] == snnz);

	for(int i = 0; i < snnz; i++)
		sortedfin >> srowinds[i];
	for(int i = 0; i < snnz; i++)
		sortedfin >> scolinds[i];
	for(int i = 0; i < snnz; i++)
		sortedfin >> svals[i];
	for(int i = 0; i < snrows; i++)
		sortedfin >> sdinds[i];
	
	for(int i = 0; i < snnz; i++)
	{
		assert(scolinds[i]-1 == mat.bcolind[i]);
		assert(svals[i] == mat.vals[i]);
	}
	for(int i = 0; i < mat.nbrows+1; i++)
		assert(srowptr[i] == mat.browptr[i]);
	for(int i = 0; i < mat.nbrows; i++)
		assert(sdinds[i] == mat.diagind[i]);

	sortedfin.close();
	return 0;
}

template class TestCSRMatrix<double>;
template class TestCSRMatrix<float>;
