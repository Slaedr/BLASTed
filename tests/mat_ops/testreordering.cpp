/** \file
 * \brief Test reordering
 * \author Aditya Kashi
 */

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

template <int bs>
int testMatrixReordering(const std::string matfile)
{
	BSRMatrix<double,int,bs> omat = constructBSRMatrixFromMatrixMarketFile<double,int,bs>(matfile);
	const int nbrows = omat.dim()/bs;

	std::vector<int> ord = generateOrdering(nbrows);

	BSRMatrix<double,int,bs> mat1(omat);

	TrivialRS<bs> rs;
	rs.setOrdering(&ord[0],&ord[0],nbrows);

	mat1.reorderScale(rs, FORWARD);
	mat1.reorderScale(rs, INVERSE);

	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 3) {
		std::cout << "Need mtx file name and block size!\n";
		std::exit(-1);
	}

	std::string matfile = argv[1];
	const int blocksize = std::stoi(argv[2]);

	switch(blocksize) {
	case(1):
		testMatrixReordering<1>(matfile);
		break;
	case(7):
		testMatrixReordering<7>(matfile);
		break;
	default:
		throw "Block size not supported!";
	}

	return 0;
}
