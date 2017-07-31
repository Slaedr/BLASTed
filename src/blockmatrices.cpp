#include "blockmatrices.hpp"

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/LU>

namespace blasted {

template <typename scalar, typename index, size_t bs>
BSRMatrixi<scalar,index,bs>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs)
	: nbrows(n_brows)
{
	browptr = new index[nbrows+1];
	bcolind = new index[brptrs[nbrows]];
	diagind = new index[nbrows];
	data = new scalar[brptrs[nbrows]*bs*bs];
	for(index i = 0; i < nbrows+1; i++)
		browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[nbrows]; i++)
		bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < nbrows; irow++) {
		for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			if(bcolind[j] == irow) {
				diagind[irow] = j;
				break;
			}
	}

	dblocks = iludata = nullptr;
}

template <typename scalar, typename index, size_t bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	delete [] data;
	delete [] bcolind;
	delete [] browptr;
	delete [] diagind;
	if(dblocks)
		delete [] dblocks;
	if(iludata)
		delete [] iludata;
	data = dblocks = iludata = nullptr;
	bcolind = browptr = diagind = nullptr;
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const size_t bsizei, const size_t bsizej, const scalar *const buffer)
{
#ifdef DEBUG
	if(bsizei != bs || bsizej != bs) {
		std::cout << "! BSRMatrix: submitBlock: The block must be " << bs " x " << bs << "!!\n";
		return;
	}
#endif
	
	bool found = false;
	constexpr size_t bs2 = bs*bs;
	for(index j = browptr[starti]; j < browptr[starti+1]; j++) {
		if(bcolind[j] == startj) {
			for(size_t k = 0; k < bs2; k++)
				data[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const size_t bsizei, const size_t bsizej, const scalar *const buffer)
{
#ifdef DEBUG
	if(bsizei != bs || bsizej != bs) {
		std::cout << "! BSRMatrix: submitDiagBlock: The block must be " << bs " x " << bs << "!!\n";
		return;
	}
	if(bsizei != bsizej) {
		std::cout << "! BSRMatrix: submitDiagBlock: The block must be square!!\n";
		return;
	}
#endif
	
	constexpr size_t bs2 = bs*bs;
	const index pos = diagind[starti];
	for(size_t k = 0; k < bs2; k++)
#pragma omp atomic update
		data[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const size_t bsizei, const size_t bsizej, const scalar *const buffer)
{
#ifdef DEBUG
	if(bsizei != bs || bsizej != bs) {
		std::cout << "! BSRMatrix: updateBlock: The block must be " << bs " x " << bs << "!!\n";
		return;
	}
#endif
	
	bool found = false;
	constexpr size_t bs2 = bs*bs;
	for(index j = browptr[starti]; j < browptr[starti+1]; j++) {
		if(bcolind[j] == startj) {
			for(size_t k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				data[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar a, const scalar *const x, scalar *const y) const
{
	const size_t bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		for(size_t j = 0; j < bs; j++)
			y[irow*bs+j] = 0;

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			const index jcol = bcolind[jj];
			
			// multiply the blocks with corresponding sub-vectors
			for(size_t i = 0; i < bs; i++)
				for(size_t j = 0; j < bs; j++)
					y[irow*bs+i] += a * data[jj*bs2 + i*bs+j] * x[jcol*bs+j];
		}
	}
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const x, 
		const scalar b, const scalar *const y, scalar *const z) const
{
	const size_t bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		for(size_t j = 0; j < bs; j++)
			z[irow*bs + j] = 0;

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			const index jcol = bcolind[jj];
			
			// multiply the blocks with corresponding sub-vectors
			for(size_t i = 0; i < bs; i++)
				for(size_t j = 0; j < bs; j++)
					z[irow*bs+i] += a * data[jj*bs2 + i*bs+j] * x[jcol*bs+j] + b*y[irow*bs+i];
		}
	}
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::precJacobiSetup()
{
	if(!dblocks) {
		dblocks = new scalar[nbrows*bs*bs];
#if DEBUG==1
		std::cout << " BSRMatrix: precJacobiSetup(): Allocating.\n";
#endif
	}

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		Eigen::Map<Eigen::Matrix<scalar,bs,bs,RowMajor>>  blk(&data[diagind[irow]]);
		Eigen::Map<Eigen::Matrix<scalar,bs,bs,RowMajor>> iblk(&dblocks[irow*bs*bs]);
		iblk = blk.inverse();
	}
}

template <typename scalar, typename index, size_t bs>
void BSRMatrix<scalar,index,bs>::precJacobiApply(const scalar *const r, scalar *const z) const
{
	constexpr size_t bs2 = bs*bs;
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		for(size_t i = 0; i < bs; i++)
			z[irow*bs+i] = 0;
		for(size_t i = 0; i < bs; i++)
			for(size_t j = 0; j < bs; j++)
				z[irow*bs+i] += dblocks[irow*bs2 + i*bs+j] * r[irow*bs+j];
	}
}

}
