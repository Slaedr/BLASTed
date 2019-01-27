/** \file
 * \brief Checks conversion from BSR to BSC
 */
#undef NDEBUG

#include <cstring>
#include "../../src/helper_algorithms.hpp"
#include "scmatrixdefs.hpp"
#include "coomatrix.hpp"

using namespace blasted;

template <int bs>
bool compare(const CRawBSRMatrix<double,int> *const mr, const RawBSCMatrix<double,int> *const mc)
{
	bool same = true;
	for(int irow = 0; irow < mr->nbrows; irow++)
	{
		for(int jr = mr->browptr[irow]; jr < mr->browptr[irow+1]; jr++)
		{
			const int col = mr->bcolind[jr];
			// Search for row irow in col
			int rowpos = -1;
			internal::inner_search(mc->browind, mc->bcolptr[col], mc->bcolptr[col+1], irow, &rowpos);
			if (rowpos < 0) {
				printf("Could not find block in BSC matrix!\n");
				return false;
			}
			// check
			for(int i = 0; i < bs*bs; i++) {
				if(mr->vals[jr*bs*bs+i] != mc->vals[rowpos*bs*bs+i])
				{
					printf("Difference = %f.\n", abs(mr->vals[jr*bs*bs+i]-mc->vals[rowpos*bs*bs+i]));
					return false;
				}
			}
		}
	}
	return same;
}

template <int bs>
int testConvertBSRToBSC(const std::string mfile)
{
	COOMatrix<double,int> coomat;
	coomat.readMatrixMarket(mfile);
	RawBSRMatrix<double,int> rmat;
	coomat.convertToBSR<bs,ColMajor>(&rmat);

	RawBSCMatrix<double,int> cmat =
		convert_BSR_to_BSC<double,int,bs>(reinterpret_cast<const CRawBSRMatrix<double,int>*>(&rmat));

	const bool pass = compare<bs>(reinterpret_cast<const CRawBSRMatrix<double,int>*>(&rmat), &cmat);

	fflush(stdout);
	assert(pass);

	destroyRawBSRMatrix(rmat);
	destroyRawBSCMatrix(cmat);
	return 0;
}

template <>
int testConvertBSRToBSC<1>(const std::string mfile)
{
	COOMatrix<double,int> coomat;
	coomat.readMatrixMarket(mfile);
	RawBSRMatrix<double,int> rmat;
	coomat.convertToCSR(&rmat);

	RawBSCMatrix<double,int> cmat =
		convert_BSR_to_BSC<double,int,1>(reinterpret_cast<const CRawBSRMatrix<double,int>*>(&rmat));

	const bool pass = compare<1>(reinterpret_cast<const CRawBSRMatrix<double,int>*>(&rmat), &cmat);

	fflush(stdout);
	assert(pass);

	destroyRawBSRMatrix(rmat);
	destroyRawBSCMatrix(cmat);
	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 3) {
		printf("Need the file name of the matrix file and the block size as arguments!");
		return -1;
	}

	const std::string matfile = argv[1];
	const int blocksize = std::stoi(argv[2]);

	switch(blocksize) {
	case 1:
		printf("Checking for block size %d\n", blocksize);
		testConvertBSRToBSC<1>(matfile);
		break;
	case 3:
		printf("Checking for block size %d\n", blocksize);
		testConvertBSRToBSC<3>(matfile);
		break;
	case 4:
		printf("Checking for block size %d\n", blocksize);
		testConvertBSRToBSC<4>(matfile);
		break;
	case 5:
		printf("Checking for block size %d\n", blocksize);
		testConvertBSRToBSC<5>(matfile);
		break;
	case 7:
		printf("Checking for block size %d\n", blocksize);
		testConvertBSRToBSC<7>(matfile);
		break;
	default:
		printf("This block size is not supported!\n");
		return -2;
	}

	return 0;
}
