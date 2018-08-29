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

	// TODO: Fill with permuted integers between 0 and N-1 (inclusive)

	return ord;
}

int testReordering(const std::string matfile)
{
	COOMatrix<double,int> cmat;
	cmat.readMatrixMarket(matfile);
}
