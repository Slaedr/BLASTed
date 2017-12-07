/** \file testcoomatrix.hpp
 * \brief Unit test specs for Coordinate matrices
 * \author Aditya Kashi
 */

#ifndef TESTCOOMATRIX_H
#define TESTCOOMATRIX_H

#include <coomatrix.hpp>
#include "testcsrmatrix.hpp"

using namespace blasted;

/// A COO matrix class for testing \ref COOMatrix
class TestCOOMatrix : public COOMatrix<double,int>
{
public:
	TestCOOMatrix();

	/// Tests basic input of a Matrix Market file
	/** \param[in] matfile Matrix market input file that contains the input matrix on which to test
	 * \param[in] sortedfile A file containing the same matrix in the following format:
	 *   - <number of rows> <number of columns> <number of non-zeros>
	 *   - <list of nrows indices at which corresponding rows start followed by nnz again>
	 *   - <row indices sorted by row index>
	 *   - <column indices sorted by column index within each row; the latter sorted by row>
	 *   - <Non-zero values sorted by row index and within each row by column index>
	 *   - <A list of nrows number of indices pointing to th position of diagonal entries of each row>
	 */
	int readCoordinateMatrix(const std::string matfile, const std::string sortedfile);

protected:
	using COOMatrix<double,int>::nnz;
	using COOMatrix<double,int>::nrows;
	using COOMatrix<double,int>::ncols;
	using COOMatrix<double,int>::entries;
	using COOMatrix<double,int>::rowptr;
};

/// Test conversion of COO to CSR
int testConvertCOOToCSR(const std::string matfile, const std::string sortedfile);
	
/// Test conversion to BSR
template <int bs, StorageOptions stor>
int testConvertCOOToBSR(const std::string matfile, const std::string sortedfile);

#endif
