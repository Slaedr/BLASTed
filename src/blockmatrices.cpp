#include "blockmatrices.hpp"

namespace blasted {

template <typename scalar, typename index, unsigned int bs>
BSRMatrixi<scalar,index,bs>::BSRMatrix(const index nbrows, const index *const blocksperrow)
{
	data = new scalar[nbrows*blocksperrow*bs];
	bcolind = new index[nbrows*blocksperrow];
	browptr = new index[nbrows+1];
	browptr[0] = 0;
	for(index i = 1; i < nbrows+1; i++) {
		browptr[i] = browptr[i-1]+blocksperrow;
	}
}

template <typename scalar, typename index, unsigned int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	delete [] data;
	delete [] bcolind;
	delete [] browptr;
}

}
