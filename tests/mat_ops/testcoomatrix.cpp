/** \file testcoomatrix.cpp
 * \brief Implementation of coordinate matrix tests
 * \author Aditya Kashi
 */

#undef NDEBUG

#ifdef DEBUG
#define __restrict__ 
#define __restrict
#endif

#include <fstream>
#include "testcoomatrix.hpp"

TestCOOMatrix::TestCOOMatrix() : COOMatrix<double,int>()
{ }

int TestCOOMatrix::readCoordinateMatrix(const std::string matfile, 
                                        const std::string sortedfile)
{
	std::ifstream matfin(matfile);
	if(!matfin) {
		std::cout << " COOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	readMatrixMarket(matfile);
	
	std::ifstream sortedfin(sortedfile);
	if(!sortedfin) {
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
	std::vector<double> svals(snnz);

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
		assert(srowinds[i]-1 == entries[i].rowind);
		assert(scolinds[i]-1 == entries[i].colind);
		assert(svals[i] == entries[i].value);
	}
	for(int i = 0; i < nrows+1; i++)
		assert(srowptr[i] == rowptr[i]);

	matfin.close();
	sortedfin.close();
	return 0;
}

template <int bs, StorageOptions stor>
int testConvertCOOToBSR(const std::string matfile, 
                        const std::string sortedfile)
{
	std::ifstream matfin(matfile);
	if(!matfin) {
		std::cout << " TestCOOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	COOMatrix<double,int> cmat;
	cmat.readMatrixMarket(matfile);
	
	std::ifstream sortedfin(sortedfile);
	if(!sortedfin) {
		std::cout << " TestCOOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	int snnz, snrows, sncols;
	sortedfin >> snrows >> sncols >> snnz;

	std::vector<int> srowinds(snnz);
	std::vector<int> srowptr(snrows+1);
	std::vector<int> scolinds(snnz);
	std::vector<double> svals(snnz*bs*bs);
	std::vector<int> sdiaginds(snrows);

	for(int i = 0; i < snrows+1; i++)
		sortedfin >> srowptr[i];

	assert(srowptr[snrows] == snnz);

	for(int i = 0; i < snnz; i++)
		sortedfin >> srowinds[i];
	for(int i = 0; i < snnz; i++)
		sortedfin >> scolinds[i];
	for(int i = 0; i < snnz*bs*bs; i++)
		sortedfin >> svals[i];
	for(int i = 0; i < snrows; i++)
		sortedfin >> sdiaginds[i];

	//RawBSRMatrix<double,int> bm;
	//cmat.convertToBSR<bs,stor>(&bm);
	const SRMatrixStorage<double,int> bm = cmat.convertToBSR<bs,stor>();

	// test

	assert(snrows == bm.nbrows);
	assert(snnz == bm.browptr[bm.nbrows]);
	assert(snnz == bm.nnzb);
	assert(snnz == bm.nbstored);

	for(int i = 0; i < snnz; i++) {
		assert(scolinds[i]-1 == bm.bcolind[i]);
	}
	for(int i = 0; i < snrows+1; i++)
		assert(srowptr[i] == bm.browptr[i]);
	for(int i = 0; i < snnz*bs*bs; i++) {
		assert(bm.vals[i] == svals[i]);
	}
	for(int i = 0; i < snrows; i++)
		assert(sdiaginds[i] == bm.diagind[i]);

	matfin.close();
	sortedfin.close();

	printf(" COO conversion to BSR SRMatrixStorage passed.\n");
	// delete [] bm.vals;
	// delete [] bm.browptr;
	// delete [] bm.bcolind;
	// delete [] bm.diagind;
	return 0;
}

template
int testConvertCOOToBSR<3,ColMajor>(const std::string matfile, const std::string sortedfile);
template
int testConvertCOOToBSR<3,RowMajor>(const std::string matfile, const std::string sortedfile);

int testConvertCOOToCSR(const std::string matfile, const std::string sortedfile)
{
	std::ifstream matfin(matfile);
	if(!matfin) {
		std::cout << " TestCOOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	COOMatrix<double,int> cmat;
	cmat.readMatrixMarket(matfile);
	
	std::ifstream sortedfin(sortedfile);
	if(!sortedfin) {
		std::cout << " TestCOOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	int snnz, snrows, sncols;
	sortedfin >> snrows >> sncols >> snnz;

	std::vector<int> srowinds(snnz);
	std::vector<int> srowptr(snrows+1);
	std::vector<int> scolinds(snnz);
	std::vector<double> svals(snnz);
	std::vector<int> sdiaginds(snrows);

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
		sortedfin >> sdiaginds[i];

	// RawBSRMatrix<double,int> bm;
	// cmat.convertToCSR(&bm);
	const SRMatrixStorage<const double, const int> bm = move_to_const(cmat.convertToCSR());

	// test

	assert(snrows == bm.nbrows);
	assert(snnz == bm.browptr[bm.nbrows]);

	for(int i = 0; i < snnz; i++) {
		assert(scolinds[i]-1 == bm.bcolind[i]);
	}
	for(int i = 0; i < snrows+1; i++)
		assert(srowptr[i] == bm.browptr[i]);
	for(int i = 0; i < snnz; i++) {
		assert(bm.vals[i] == svals[i]);
	}
	for(int i = 0; i < snrows; i++)
		assert(sdiaginds[i] == bm.diagind[i]);

	matfin.close();
	sortedfin.close();

	printf(" COO conversion to CSR SRMatrixStorage passed.\n");

	// delete [] bm.vals;
	// delete [] bm.browptr;
	// delete [] bm.bcolind;
	// delete [] bm.diagind;
	return 0;
}

