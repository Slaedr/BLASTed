#undef NDEBUG

#include "coomatrix.hpp"
#include "blockmatrices.hpp"
#include "reorderingscaling.hpp"

using namespace blasted;

int main(int argc, char *argv[])
{
	if(argc < 2) {
		std::cout << "Need mtx file name.\n";
		std::exit(-1);
	}

	std::string matfile = argv[1];

	BSRMatrix<double,int,1> omat = constructBSRMatrixFromMatrixMarketFile<double,int,1>(matfile);
	//const int nbrows = omat.dim();

	MC64 mc64;
	omat.computeOrderingScaling(mc64);

	BSRMatrix<double,int,1> mat1(omat);

	mat1.reorderScale(mc64, FORWARD);
	mat1.reorderScale(mc64, INVERSE);

	const std::array<bool,5> reseq = omat.isEqual(mat1, 1e-9);

	assert(reseq[0]);
	assert(reseq[1]);
	assert(reseq[2]);
	assert(reseq[3]);
	assert(reseq[4]);

	return 0;
}
