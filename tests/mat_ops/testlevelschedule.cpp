#undef NDEBUG

#include "coomatrix.hpp"
#include "blockmatrices.hpp"
#include "levelschedule.hpp"

using namespace blasted;

template <int bs>
int test_levelschedule(const std::string matfile)
{
	COOMatrix<double,int> coo;
	coo.readMatrixMarket(matfile);

	SRMatrixStorage<const double,const int> smat =
		move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(coo, "colmajor"));
	CRawBSRMatrix<double,int> omat(&smat.browptr[0], &smat.bcolind[0], &smat.vals[0],
	                               &smat.diagind[0], &smat.browendptr[0], smat.nbrows,
	                               smat.nnzb, smat.nbstored);

	std::cout << "Number of block rows = " << omat.nbrows << std::endl;

	const std::vector<int> levels = computeLevels(&omat);

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
	// case 3:
	// 	res = test_levelschedule<3>(matfile);
	// 	break;
	case 4:
		res = test_levelschedule<4>(matfile);
		break;
	// case 5:
	// 	res = test_levelschedule<5>(matfile);
	// 	break;
	// case 6:
	// 	res = test_levelschedule<6>(matfile);
	// 	break;
	// case 7:
	// 	res = test_levelschedule<7>(matfile);
	// 	break;
	default:
		printf("Block size not available!");
	}

	return res;
}
