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

	std::array<bool,5> reseq = omat.isEqual(mat1);

	assert(reseq[0]);
	assert(reseq[1]);
	assert(reseq[2]);
	assert(reseq[3]);
	assert(reseq[4]);

	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 4) {
		std::cout << "Need mtx file name and block size!\n";
		std::exit(-1);
	}

	std::string matfile = argv[1];
	const int blocksize = std::stoi(argv[2]);
	const std::string rsdir = argv[3];

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

	return 0;
}
