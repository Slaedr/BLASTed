#include "blockmatrices.hpp"

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/LU>

namespace blasted {

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const unsigned short n_buildsweeps, const unsigned short n_applysweeps)
	: nbrows{n_brows}, dblocks(nullptr), iludata(nullptr),
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{
	constexpr int bs2 = bs*bs;
	browptr = new index[nbrows+1];
	bcolind = new index[brptrs[nbrows]];
	diagind = new index[nbrows];
	data.resize(brptrs[nbrows]*bs, bs);
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
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	delete [] bcolind;
	delete [] browptr;
	delete [] diagind;
	bcolind = browptr = diagind = nullptr;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const long param1, const long param2) 
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
				data.data()[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const scalar *const buffer)
{
	constexpr int bs2 = bs*bs;
	const index startr = starti/bs;
	const index pos = diagind[startr];
	for(int k = 0; k < bs2; k++)
#pragma omp atomic update
		data.data()[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const long param1, const long param2)
{
#ifdef DEBUG
	if(bsizei != bs || bsizej != bs) {
		std::cout << "! BSRMatrix: updateBlock: The block must be " << bs " x " << bs << "!!\n";
		return;
	}
#endif
	
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				data.data()[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict__ yy) const
{
	constexpr int bs2 = bs*bs;
	Eigen::Map<const Vector> x(xx, nbrows*bs);
	Eigen::Map<Vector> y(yy, nbrows*bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		y.segment<bs>(irow*bs) = Matrix<scalar,bs,1>::Zero();

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = bcolind[jj];
			y.segment<bs>(irow*bs).noalias() 
				+= a * data.block<bs,bs>(jj*bs,0) * x.segment<bs>(jcol*bs);
			
			/*for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					y[irow*bs+i] += a * data[jj*bs2 + i*bs+j] * x[jcol*bs+j];*/
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	constexpr int bs2 = bs*bs;
	Eigen::Map<const Vector> x(xx, nbrows*bs);
	Eigen::Map<const Vector> y(yy, nbrows*bs);
	Eigen::Map<Vector> z(zz, nbrows*bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		z.segment<bs>(irow*bs) = b * y.segment<bs>(irow*bs);

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			const index jcol = bcolind[jj];
			z.segment<bs>(irow*bs).noalias() += 
				a * data.block<bs,bs>(jj*bs,0) * x.segment<bs>(jcol*bs);
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiSetup()
{
	if(dblocks.size() <= 0) {
		dblocks.resize(nbrows*bs,bs);
#if DEBUG==1
		std::cout << " BSRMatrix: precJacobiSetup(): Allocating.\n";
#endif
	}

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		dblocks.block<bs,bs>(irow*bs,0) = data.block<bs,bs>(diagind[irow]*bs).inverse();
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict__ zz) const
{
	Eigen::Map<const Vector> r(rr, nbrows*bs);
	Eigen::Map<Vector> z(zz, nbrows*bs);
	constexpr int bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		z.segment<bs>(irow*bs).noalias() = dblocks.block<bs,bs>(irow*bs,0) * r.segment<bs>(irow*bs);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::allocTempVector()
{
	ytemp.resize(nbrows*bs);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict__ zz) const
{
	Eigen::Map<const Vector> r(rr, nbrows*bs);
	Eigen::Map<Vector> z(zz, nbrows*bs);
	constexpr int bs2 = bs*bs;

	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

			for(index jj = browptr[irow]; jj < diagind[irow]; jj++)
				inter += data.block<bs,bs>(jj*bs,0)*ytemp.segment<bs>(bcolind[jj]);

			ytemp.segment<bs>(irow) = dblocks.block<bs,bs>(irow*bs,0) 
			                                          * (r.segment<bs>(irow) - inter);
		}
	}

	for(unsigned short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
			
			// compute U z
			for(index jj = diagind[irow]; jj < browptr[irow+1]; jj++)
				inter += data.block<bs,bs>(jj*bs,0) * z.segment<bs>(bcolind[jj]*bs);

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			z.segment<bs>(irow*bs) = dblocks.block<bs,bs>(irow*bs2,0) 
				* ( data.block<bs,bs>(diagind[irow]*bs,0)*ytemp.segment<bs>(irow*bs) - inter );
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUSetup()
{
	// TODO
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUApply(const scalar *const r, 
                                              scalar *const __restrict__ z) const
{
	// TODO
}

}

