/** \file testcsrmatrix.cpp
 * \brief Implementation of tests for CSR matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <vector>
#include <fstream>
#include <float.h>
#include <coomatrix.hpp>
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

template <typename scalar>
int TestCSRMatrix<scalar>::writeCOO(const std::string outfile)
{
	// write
	std::ofstream fout(outfile);
	if(!fout) {
		return -1;
	}

	fout << mat.nbrows << " " << mat.nbrows << " " << mat.browptr[mat.nbrows] << '\n';
	for(int irow = 0; irow < mat.nbrows; irow++)
	{
		for(int j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			fout << irow+1 << " " << mat.bcolind[j]+1 << " " << mat.vals[j] << '\n';
	}
	fout.close();
	return 0;
}

template class TestCSRMatrix<double>;

int testCSRMatMult(const std::string type,
		const std::string matfile, const std::string xvec, const std::string prodvec)
{
	RawBSRMatrix<double,int> rm;
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);
	coom.convertToCSR(&rm);

	const std::vector<double> x = readDenseMatrixMarket<double>(xvec);
	const std::vector<double> ans = readDenseMatrixMarket<double>(prodvec);
	std::vector<double> y(rm.nbrows);
	
	AbstractLinearOperator<double,int>* testmat = nullptr;
	if(type == "view") {
		testmat = new CSRMatrixView<double,int>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind,1,1);
	}
	else
		testmat = new BSRMatrix<double,int,1>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind,1,1);
	

	testmat->apply(x.data(), y.data());

	for(int i = 0; i < rm.nbrows; i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	delete testmat;
	destroyRawBSRMatrix(rm);

	return 0;
}

