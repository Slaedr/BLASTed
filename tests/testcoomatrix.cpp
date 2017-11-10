/** \file testcoomatrix.cpp
 * \brief Implementation of coordinate matrix tests
 * \author Aditya Kashi
 */

#ifdef DEBUG
#define __restrict__ 
#define __restrict
#endif

#include "testcoomatrix.hpp"

TestCOOMatrix::TestCOOMatrix() : COOMatrix<double,int>()
{ }

int TestCOOMatrix::readCoordinateMatrix(const std::string matfile, 
		const std::string sortedfile) const
{
	std::ifstream matfin(matfile);
	if(!fin) {
		std::cout << " COOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	readMatrixMarket(matfile);
	
	std::ifstream sortedfin(sortedfile);
	if(!fin) {
		std::cout << " COOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	int snnz, snrows, sncols;
	sortedfin >> snrows >> sncols >> snnz;

	assert(snnz == nnz);
	assert(snrows == nrows);
	assert(sncols == ncols);

	std::vector<int> srowinds(snnz);
	std::vector<int> srowptr(snrows+1);
	std::vector<int> scolinds(snnz);
	std::vector<int> svals(snnz);

	for(int i = 0; i < snrows+1; i++)
		sortedfin >> srowptr[i];

	assert(srowptr[snrows] == snnz);

	for(int i = 0; i < snnz; i++)
		sortedfin >> srowinds[i];
	for(int i = 0; i < snnz; i++)
		sortedfin >> scolinds[i];
	for(int i = 0; i < snnz; i++)
		sortedfin >> svals[i];

	// test
	for(int i = 0; i < nnz; i++)
	{
		assert(srowinds[i] == entries[i].rowind);
		assert(scolinds[i] == entries[i].colind);
		assert(svals[i] == entries[i].value);
	}
	for(int i = 0; i < nrows+1; i++)
		assert(srowptr[i] == rowptr[i]);

	matfine.close();
	sortedfin.close();
	return 0;
}
