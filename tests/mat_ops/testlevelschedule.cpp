#undef NDEBUG

#include "coomatrix.hpp"
#include "blockmatrices.hpp"
#include "levelschedule.hpp"

using namespace blasted;

template <int bs>
int test_levelschedule(const std::string matfile)
{
	RawBSRMatrix<double,int> omat;
	COOMatrix<double,int> coo;
	coo.readMatrixMarket(matfile);
	coo.convertToBSR<bs,ColMajor>(&omat);
	std::cout << "Number of block rows = " << omat.nbrows << std::endl;

	const std::vector<int> levels
		= computeLevels(reinterpret_cast<const CRawBSRMatrix<double,int>&>(omat));

	for(size_t lvl = 0; lvl < levels.size()-1; lvl++)
	{
		for(int inode = levels[lvl]; inode < levels[lvl+1]; inode++)
		{
			for(int jj = omat.browptr[inode]; jj < omat.browptr[inode+1]; jj++)
			{
				const int colind = omat.bcolind[jj];
				for(int jnode = inode+1; jnode < levels[lvl+1]; jnode++)
					if(colind == jnode)
						throw "Dependence in level!";
			}
		}
	}

	destroyRawBSRMatrix(omat);
	return 0;
}

template <>
int test_levelschedule<1>(const std::string matfile)
{
	RawBSRMatrix<double,int> omat;
	COOMatrix<double,int> coo;
	coo.readMatrixMarket(matfile);
	coo.convertToCSR(&omat);
	std::cout << "Number of block rows = " << omat.nbrows << std::endl;

	const std::vector<int> levels
		= computeLevels(reinterpret_cast<const CRawBSRMatrix<double,int>&>(omat));

	for(size_t lvl = 0; lvl < levels.size()-1; lvl++)
	{
		for(int inode = levels[lvl]; inode < levels[lvl+1]; inode++)
		{
			for(int jj = omat.browptr[inode]; jj < omat.browptr[inode+1]; jj++)
			{
				const int colind = omat.bcolind[jj];
				for(int jnode = inode+1; jnode < levels[lvl+1]; jnode++)
					if(colind == jnode)
						throw "Dependence in level!";
			}
		}
	}

	destroyRawBSRMatrix(omat);
	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 2) {
		std::cout << "Need mtx file name and block size\n";
		std::exit(-1);
	}

	const std::string matfile = argv[1];
	const int blocksize = std::stoi(argv[2]);

	int res = -1;
	switch(blocksize) {
	case 1:
		res = test_levelschedule<1>(matfile);
		break;
	case 3:
		res = test_levelschedule<3>(matfile);
		break;
	case 4:
		res = test_levelschedule<4>(matfile);
		break;
	case 5:
		res = test_levelschedule<5>(matfile);
		break;
	case 6:
		res = test_levelschedule<6>(matfile);
		break;
	case 7:
		res = test_levelschedule<7>(matfile);
		break;
	default:
		printf("Block size not available!");
	}

	return res;
}
