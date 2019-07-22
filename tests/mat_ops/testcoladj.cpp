/** \file
 * \brief Test column adjacency computation
 */

#undef NDEBUG
#include <stdexcept>
#include "coomatrix.hpp"
#include "../src/adjacency.hpp"

using namespace blasted;

int main(int argc, char *argv[])
{
	if(argc < 2)
		throw std::runtime_error("Need matrix file!");

	const std::string mfile = argv[1];

	COOMatrix<double,int> coomat;
	coomat.readMatrixMarket(mfile);

	const SRMatrixStorage<double,int> rmat = coomat.convertToCSR();

	const ColumnAdjacency<double,int> ca(rmat);
	const std::vector<int>& cnzrow = ca.col_nonzero_rows();
	const std::vector<int>& cnzloc = ca.col_nonzero_locations();
	const std::vector<int>& cptr = ca.col_pointers();

	// test
	for(int jcol = 0; jcol < rmat.nbrows; jcol++) {
		for(int ii = cptr[jcol]; ii < cptr[jcol+1]; ii++)
		{
			const int rowind = cnzrow[ii];
			const int pos = cnzloc[ii];
			assert(rmat.bcolind[pos] == jcol);
			assert(pos >= rmat.browptr[rowind]);
			assert(pos < rmat.browptr[rowind+1]);
		}
	}
	return 0;
}
