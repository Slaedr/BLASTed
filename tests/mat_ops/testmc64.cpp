#undef NDEBUG

#include "blockmatrices.hpp"
#include "reorderingscaling.hpp"

int main()
{
	if(argc < 2) {
		std::cout << "Need mtx file name.\n";
		std::exit(-1);
	}

	std::string matfile = argv[1];

	BSRMatrix<double,int,bs> omat = constructBSRMatrixFromMatrixMarketFile<double,int,bs>(matfile);
	const int nbrows = omat.dim()/bs;

	MC64 mc64;

	mat1.reorderScale(rs, FORWARD);
	mat1.reorderScale(rs, INVERSE);

	return 0;
}
