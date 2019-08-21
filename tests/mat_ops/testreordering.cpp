/** \file
 * \brief Test reordering
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <cassert>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "coomatrix.hpp"
#include "reorderingscaling.hpp"

using namespace blasted;

template <int bs>
class TrivialRS : public ReorderingScaling<double,int,bs>
{
public:
	TrivialRS() { }
	void compute(const CRawBSRMatrix<double,int>& mat) { }
};

std::vector<int> generateOrdering(const int N)
{
	std::vector<int> ord(N);

	for(int i = 0; i < N; i++)
		ord[i] = N-i-1;
	std::swap(ord[N/4], ord[3*N/4]);
	std::swap(ord[N/8], ord[7*N/8]);

	return ord;
}

template <int bs>
int testVectorReordering(const std::string vecfile, const std::string rsdir)
{
	const device_vector<double> ovec = readDenseMatrixMarket<double>(vecfile);
	const int nbrows = static_cast<int>(ovec.size())/bs;

	std::vector<int> ord1 = generateOrdering(nbrows);
	std::vector<int> ord2(nbrows);
	for(int i = 0; i < nbrows; i++)
		ord2[i] = i;

	std::vector<double> vec1(ovec.size());
	std::copy(ovec.begin(), ovec.end(), vec1.begin());

	TrivialRS<bs> rs;
	if(rsdir == "row")
		rs.setOrdering(&ord1[0],&ord2[0],nbrows);
	else if(rsdir == "column")
		rs.setOrdering(&ord2[0],&ord1[0],nbrows);
	else if(rsdir == "both")
		rs.setOrdering(&ord1[0],&ord1[0],nbrows);

	const RSApplyDir dir = rsdir == "row" ? ROW : COLUMN;

	rs.applyOrdering(&vec1[0], FORWARD, dir);
	rs.applyOrdering(&vec1[0], INVERSE, dir);

	for(int i = 0; i < nbrows*bs; i++)
		assert(vec1[i] == ovec[i]);

	return 0;
}

/** Verify an ordering by applying it followed by applying its inverse
 * and checking if the result is equal to the original matrix.
 */
template <int bs>
int testMatrixReordering(const std::string matfile, const std::string rsdir)
{
	BSRMatrix<double,int,bs> omat = constructBSRMatrixFromMatrixMarketFile<double,int,bs>(matfile);
	const int nbrows = omat.dim()/bs;

	std::vector<int> ord1 = generateOrdering(nbrows);
	std::vector<int> ord2(nbrows);
	for(int i = 0; i < nbrows; i++)
		ord2[i] = i;

	BSRMatrix<double,int,bs> mat1(omat);

	TrivialRS<bs> rs;
	if(rsdir == "row")
		rs.setOrdering(&ord1[0],&ord2[0],nbrows);
	else if(rsdir == "column")
		rs.setOrdering(&ord2[0],&ord1[0],nbrows);
	else if(rsdir == "both")
		rs.setOrdering(&ord1[0],&ord1[0],nbrows);

	mat1.reorderScale(rs, FORWARD);
	mat1.reorderScale(rs, INVERSE);

	std::array<bool,5> reseq = omat.isEqual(mat1, 1e-16);

	assert(reseq[0]);
	assert(reseq[1]);
	assert(reseq[2]);
	assert(reseq[3]);
	assert(reseq[4]);

	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 5) {
		std::cout << "Need mtx file name, block size, transform direction and matrix/vector.\n";
		std::exit(-1);
	}

	std::string matfile = argv[1];
	const int blocksize = std::stoi(argv[2]);
	const std::string rsdir = argv[3];
	const std::string matorvec = argv[4];

	if(matorvec == "matrix")
		switch(blocksize) {
		case(1):
			testMatrixReordering<1>(matfile, rsdir);
			break;
		case(7):
			testMatrixReordering<7>(matfile, rsdir);
			break;
		default:
			throw "Block size not supported!";
		}
	else {
		switch(blocksize) {
		case(1):
			testVectorReordering<1>(matfile, rsdir);
			break;
		case(7):
			testVectorReordering<7>(matfile, rsdir);
			break;
		default:
			throw "Block size not supported!";
		}
	}

	return 0;
}
