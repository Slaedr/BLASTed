/** \file
 * \brief Test reordering
 * \author Aditya Kashi
 */

#include <vector>
#include <string>
#include "coomatrix.hpp"
#include "reordering.hpp"

using namespace fvens;

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
int testReordering(const std::string matfile)
{
	COOMatrix<double,int> cmat;
	cmat.readMatrixMarket(matfile);
	RawBSRMatrix<double,int> bmat;
	cmat.template convertToBSR<bs,ColMajor>(&bmat);

	std::vector<int> ord = generateOrdering(bmat.nbrows);

	RawBSRMatrix<double,int> mat1, mat2;

	TrivialRS<bs> rs;
}
