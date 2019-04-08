/** \file
 * \brief Tests for sparse-column storage
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <fstream>
#include "bsr/scmatrixdefs.hpp"
#include "coomatrix.hpp"

using namespace blasted;

/// Compares a CSC matrix with one read from a file in CSC format
/** The function returns iff the two matrices are the same.
 */
int checkCSCMatrix(const CRawBSCMatrix<double,int>& cmat, const std::string cfile)
{
	std::ifstream infile(cfile);
	int dumi;
	RawBSCMatrix<double,int> fmat;

	infile >> fmat.nbcols;
	assert(fmat.nbcols == cmat.nbcols);
	infile >> dumi;
	assert(fmat.nbcols == dumi);
	fmat.bcolptr = new int[fmat.nbcols+1];
	int fnnz; infile >> fnnz;
	for(int i = 0; i < fmat.nbcols+1; i++) {
		infile >> fmat.bcolptr[i];
		assert(fmat.bcolptr[i] == cmat.bcolptr[i]);
	}
	assert(fmat.bcolptr[fmat.nbcols] == fnnz);

	fmat.browind = new int[fnnz];
	fmat.vals = new double[fnnz];
	fmat.diagind = new int[fmat.nbcols];

	for(int i = 0; i < fnnz; i++) {
		infile >> fmat.browind[i];
		assert(fmat.browind[i] == cmat.browind[i]);
	}
	for(int i = 0; i < fnnz; i++) {
		infile >> fmat.vals[i];
		assert(fmat.vals[i] == cmat.vals[i]);
	}
	for(int i = 0; i < fmat.nbcols; i++) {
		infile >> fmat.diagind[i];
		assert(fmat.diagind[i] == cmat.diagind[i]);
	}

	infile.close();

	destroyRawBSCMatrix(fmat);
	return 0;
}

/// Reads a matrix in MTX format, converts it to CSR, then tests the conversion of that to CSC
/** \param mfile File containing the matrix in COO format
 * \param solnfile File containing the same matrix in CSC format
 */
int testConvertCSRToCSC(const std::string mfile, const std::string solnfile)
{
	COOMatrix<double,int> coomat;
	coomat.readMatrixMarket(mfile);
	RawBSRMatrix<double,int> rmat;
	coomat.convertToCSR(&rmat);

	CRawBSCMatrix<double,int> cmat;
	convert_BSR_to_BSC<double,int,1>(reinterpret_cast<const CRawBSRMatrix<double,int>*>(&rmat), &cmat);

	int ierr = checkCSCMatrix(cmat, solnfile);

	destroyRawBSRMatrix(rmat);
	destroyRawBSCMatrix(cmat);
	return ierr;
}

int main(int argc, char *argv[])
{
	assert(argc >= 3);
	const std::string mfile = argv[1];
	const std::string solnfile = argv[2];

	return testConvertCSRToCSC(mfile, solnfile);
}
