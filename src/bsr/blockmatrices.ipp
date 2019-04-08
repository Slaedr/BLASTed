/** \file blockmatrices.ipp
 * \brief Implementation of block matrix methods,
 *   including the specialized set of methods for block size 1.
 * \author Aditya Kashi
 * \date 2017-08
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <type_traits>
#include <array>
#include <Eigen/Core>

#include <bsr/blockmatrices.hpp>
#include "matvecs.hpp"

namespace blasted {

/// Checks equality of two BSR/CSR matrices of the same block size
/** \param[in] tol Tolerance for non-zero values
 * \return Returns 5 booleans corresponding to equality of, in order,
 * - Number of block-rows
 * - Block-row pointers (including total number of non-zero blocks)
 * - Block-column indices
 * - Non-zero values
 * - Positions of diaginal blocks
 *
 * \warning May fail if the two arguments have different block sizes.
 */
template<typename scalar, typename index, int bs>
static std::array<bool,5> areEqual(const CRawBSRMatrix<scalar,index> *const mat1,
                                   const CRawBSRMatrix<scalar,index> *const mat2,
                                   const scalar tol)
{
	std::array<bool,5> isar;
	for(int j = 0; j < 5; j++)
		isar[j] = true;

	if(mat1->nbrows != mat2->nbrows) {
		isar[0] = false;
		return isar;
	}

	if(mat1->browptr[mat1->nbrows] != mat2->browptr[mat2->nbrows]) {
		isar[1] = false;
		return isar;
	}

	for(index i = 0; i < mat1->nbrows; i++)
	{
		if(mat1->browptr[i] != mat2->browptr[i])
			isar[1] = false;
		if(mat1->diagind[i] != mat2->diagind[i])
			isar[4] = false;
	}

	for(index jj = 0; jj < mat1->browptr[mat1->nbrows]; jj++)
	{
		if(mat1->bcolind[jj] != mat2->bcolind[jj])
			isar[2] = false;
		for(int k = 0; k < bs*bs; k++)
			if(std::abs(mat1->vals[jj*bs*bs+k] - mat2->vals[jj*bs*bs+k]) > tol) {
				isar[3] = false;
				// std::cout << "Difference is "
				//           << std::abs(mat1->vals[jj*bs*bs+k] - mat2->vals[jj*bs*bs+k]) << std::endl;
			}
	}

	return isar;
}

template<typename scalar, typename index>
SRMatrixView<scalar,index>::SRMatrixView(const index n_brows, 
		const index *const brptrs, const index *const bcinds, const scalar *const values, 
		const index *const diaginds,const StorageType storagetype) 
	: MatrixView<scalar,index>(storagetype),
	  mat{brptrs, bcinds, values, diaginds, n_brows}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
BSRMatrixView<scalar,index,bs,stor>::BSRMatrixView(const index n_brows, const index *const brptrs,
                                                   const index *const bcinds,
                                                   const scalar *const values,
                                                   const index *const diaginds)
	: SRMatrixView<scalar,index>(n_brows, brptrs, bcinds, values, diaginds, VIEWBSR)
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::wrap(const index n_brows, const index *const brptrs,
	const index *const bcinds, const scalar *const values, const index *const dinds)
{
	mat.nbrows = n_brows;
	mat.browptr = brptrs;
	mat.bcolind = bcinds;
	mat.vals = values;
	mat.diagind = dinds;
}

template <typename scalar, typename index, int bs, StorageOptions stor>
BSRMatrixView<scalar, index, bs,stor>::~BSRMatrixView()
{
	mat.browptr = nullptr;
	mat.bcolind = nullptr;
	mat.vals = nullptr;
	mat.diagind = nullptr;
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::apply(const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	bsr_matrix_apply<scalar,index,bs,stor>(&mat, xx, yy);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::gemv3(const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	bsr_gemv3<scalar,index,bs,stor>(&mat, a, xx, b, yy, zz);
}

template <typename scalar, typename index>
CSRMatrixView<scalar,index>::CSRMatrixView(const index nrows, const index *const brptrs,
                                           const index *const bcinds, const scalar *const values,
                                           const index *const diaginds)
	: SRMatrixView<scalar,index>(nrows, brptrs, bcinds, values, diaginds, VIEWCSR)
{ }

template <typename scalar, typename index>
CSRMatrixView<scalar,index>::~CSRMatrixView()
{ 
	mat.browptr = nullptr;
	mat.bcolind = nullptr;
	mat.vals = nullptr;
	mat.diagind = nullptr;
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::wrap(const index n_brows, const index *const brptrs,
	const index *const bcinds, const scalar *const values, const index *const dinds)
{
	mat.nbrows = n_brows;
	mat.browptr = brptrs;
	mat.bcolind = bcinds;
	mat.vals = values;
	mat.diagind = dinds;
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::apply(const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	csr_matrix_apply(&mat, xx, yy);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	csr_gemv3(&mat, a, xx, b, yy, zz);
}

////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix()
	: AbstractMatrix<scalar,index>(BSR), mat{nullptr, nullptr, nullptr, nullptr, 0}
{
	std::cout << "BSRMatrix: Initialized matrix without allocation.\n";
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows,
                                      const index *const bcinds, const index *const brptrs)
	: AbstractMatrix<scalar,index>(BSR), owner{true}, 
	mat{nullptr, nullptr, nullptr, nullptr, n_brows}
{
	constexpr int bs2 = bs*bs;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]*bs2];
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) 
	{
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	std::cout << "BSRMatrix: Setup with matrix with " << mat.nbrows << " block-rows\n";
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows, index *const brptrs, index *const bcinds,
                                      scalar *const values, index *const diaginds)
	: AbstractMatrix<scalar,index>(BSR), owner{false},
	  mat{brptrs, bcinds, values, diaginds, n_brows}
{ }

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(RawBSRMatrix<scalar,index>& rmat)
	: AbstractMatrix<scalar,index>(BSR), owner{true}, mat{rmat.browptr, rmat.bcolind, rmat.vals,
			                                                  rmat.diagind, rmat.nbrows}
{
	rmat.nbrows=0;
	rmat.browptr = rmat.bcolind = rmat.diagind = nullptr; rmat.vals = nullptr;
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const BSRMatrix<scalar,index,bs>& other)
	: AbstractMatrix<scalar,index>(BSR), owner{true}
{
	constexpr int bs2 = bs*bs;
	mat.nbrows = other.mat.nbrows;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[other.mat.browptr[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[other.mat.browptr[mat.nbrows]*bs2];

	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = other.mat.browptr[i];

	for(index i = 0; i < mat.browptr[mat.nbrows]; i++) {
		mat.bcolind[i] = other.mat.bcolind[i];
		for(int k = 0; k < bs2; k++)
			mat.vals[i*bs2 + k] = other.mat.vals[i*bs2+k];
	}

	for(index irow = 0; irow < mat.nbrows; irow++) 
		mat.diagind[irow] = other.mat.diagind[irow];
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	if(owner) {
		delete [] mat.vals;
		delete [] mat.bcolind;
		delete [] mat.browptr;
		delete [] mat.diagind;
	}
	mat.bcolind = mat.browptr = mat.diagind = nullptr; mat.vals = nullptr;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setStructure(const index n_brows,
		const index *const bcinds, const index *const brptrs)
{
	delete [] mat.vals;
	delete [] mat.bcolind;
	delete [] mat.browptr;
	delete [] mat.diagind;

	mat.nbrows = n_brows;
	constexpr int bs2 = bs*bs;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]*bs2];
	owner = true;
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) 
	{
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	std::cout << "BSRMatrix:  Allocated storage for matrix with " << mat.nbrows << " block-rows.\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setAllZero()
{
	const index nnz = mat.browptr[mat.nbrows]*bs*bs;
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < nnz; i++)
		mat.vals[i] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setDiagZero()
{
	constexpr unsigned int bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
#pragma omp simd
		for(index jj = mat.diagind[irow]*bs2; jj < (index)((mat.diagind[irow]+1)*bs2); jj++)
			mat.vals[jj] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2) 
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = mat.browptr[startr]; j < mat.browptr[startr+1]; j++) {
		if(mat.bcolind[j] == startc) 
		{
			for(int k = 0; k < bs2; k++)
				mat.vals[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index param)
{
	constexpr int bs2 = bs*bs;
	const index startr = starti/bs;
	const index pos = mat.diagind[startr];
	for(int k = 0; k < bs2; k++)
#pragma omp atomic update
		mat.vals[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2)
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;

	for(index j = mat.browptr[startr]; j < mat.browptr[startr+1]; j++) {
		if(mat.bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				mat.vals[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::scaleAll(const scalar factor)
{
#pragma omp parallel for default(shared)
	for(index iz = 0; iz < mat.browptr[mat.nbrows]; iz++)
	{
#pragma omp simd
		for(index k = 0; k < bs*bs; k++)
			mat.vals[iz*bs*bs + k] *= factor;
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	bsr_matrix_apply<scalar,index,bs,RowMajor>(
			reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat), xx, yy);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	bsr_gemv3<scalar,index,bs,RowMajor>(
			reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat), 
			a, xx, b, yy, zz);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::computeOrderingScaling(ReorderingScaling<scalar,index,bs>& rs) const
{
	const CRawBSRMatrix<scalar,index>* cmat
		= reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat);
	rs.compute(*cmat);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::reorderScale(const ReorderingScaling<scalar,index,bs>& rs,
                                              const RSApplyMode mode)
{
	if(mode == FORWARD) {
		rs.applyScaling(mat,mode);
		rs.applyOrdering(mat,mode);
	}
	else {
		rs.applyOrdering(mat,mode);
		rs.applyScaling(mat,mode);
	}
}

template <typename scalar, typename index, int bs>
std::array<bool,5> BSRMatrix<scalar,index,bs>::isEqual(const BSRMatrix<scalar,index,bs>& other,
                                                       const scalar tol) const
{
	return areEqual<scalar,index,bs>(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat),
	                                 reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&other.mat),
	                                 tol);
}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar, typename index>
inline
BSRMatrix<scalar,index,1>::BSRMatrix()
	: AbstractMatrix<scalar,index>(CSR), mat{nullptr,nullptr,nullptr,nullptr,0}
{
#ifdef DEBUG
	std::cout << "BSRMatrix<1>: Initialized CSR matrix.";
#endif
}

template <typename scalar, typename index>
inline
BSRMatrix<scalar,index,1>::BSRMatrix(const index n_brows,
                                     const index *const bcinds, const index *const brptrs)
	: AbstractMatrix<scalar,index>(CSR), owner{true},
	mat{nullptr,nullptr,nullptr,nullptr,n_brows}
{
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]];
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) {
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	
	std::cout << "BSRMatrix<1>: Set up CSR matrix with " << mat.nbrows << " rows\n";
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(const index nrows, index *const brptrs,
                                     index *const bcinds, scalar *const values, index *const diaginds)
	: AbstractMatrix<scalar,index>(CSR), owner{false}, mat{brptrs,bcinds,values,diaginds,nrows}
{ }

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(RawBSRMatrix<scalar,index>& rmat)
	: AbstractMatrix<scalar,index>(CSR), owner{true}, mat{rmat.browptr, rmat.bcolind, rmat.vals,
			                                                  rmat.diagind, rmat.nbrows}
{
	rmat.nbrows=0;
	rmat.browptr = rmat.bcolind = rmat.diagind = nullptr; rmat.vals = nullptr;
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(const BSRMatrix<scalar,index,1>& other)
	: AbstractMatrix<scalar,index>(CSR), owner{true}
{
	mat.nbrows = other.mat.nbrows;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[other.mat.browptr[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[other.mat.browptr[mat.nbrows]];

	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = other.mat.browptr[i];
	for(index i = 0; i < mat.browptr[mat.nbrows]; i++) {
		mat.bcolind[i] = other.mat.bcolind[i];
		mat.vals[i] = other.mat.vals[i];
	}

	for(index irow = 0; irow < mat.nbrows; irow++) 
		mat.diagind[irow] = other.mat.diagind[irow];
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::~BSRMatrix()
{
	if(owner) {
		delete [] mat.vals;
		delete [] mat.bcolind;
		delete [] mat.browptr;
		delete [] mat.diagind;
	}

	mat.bcolind = mat.browptr = mat.diagind = nullptr;
	mat.vals = nullptr;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setStructure(const index n_brows,
		const index *const bcinds, const index *const brptrs)
{
	delete [] mat.vals;
	delete [] mat.bcolind;
	delete [] mat.browptr;
	delete [] mat.diagind;

	mat.nbrows = n_brows;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]];
	owner = true;
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) {
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	
	std::cout << "BSRMatrix<1>: Set up CSR matrix with " << mat.nbrows << " rows.\n";
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setAllZero()
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.browptr[mat.nbrows]; i++)
		mat.vals[i] = 0;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setDiagZero()
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
		mat.vals[mat.diagind[irow]] = 0;
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			if(mat.bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(mat.bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: submitBlock: Invalid block!!\n";
#endif
			mat.vals[jj] = buffer[locrow*bsj+k];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index bs)
{
	// update the block, row-wise
	for(index irow = starti; irow < starti+bs; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.diagind[irow]-locrow; jj < mat.diagind[irow]-locrow+bs; jj++)
		{
			index loccol = jj-mat.diagind[irow]+locrow;
#pragma omp atomic update
			mat.vals[jj] += buffer[locrow*bs+loccol];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			if(mat.bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(mat.bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: updateBlock: Invalid block!!\n";
#endif
#pragma omp atomic update
			mat.vals[jj] += buffer[locrow*bsi+k];
			k++;
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::scaleAll(const scalar factor)
{
#pragma omp parallel for simd default(shared)
	for(index iz = 0; iz < mat.browptr[mat.nbrows]; iz++)
	{
		mat.vals[iz] *= factor;
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::apply(const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	csr_matrix_apply(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat), xx, yy);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	csr_gemv3(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat), a, xx, b, yy, zz);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::computeOrderingScaling(ReorderingScaling<scalar,index,1>& rs) const
{
	const CRawBSRMatrix<scalar,index>* cmat
		= reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat);
	rs.compute(*cmat);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::reorderScale(const ReorderingScaling<scalar,index,1>& rs,
                                             const RSApplyMode mode)
{
	if(mode == FORWARD) {
		rs.applyScaling(mat,mode);
		rs.applyOrdering(mat,mode);
	}
	else {
		rs.applyOrdering(mat,mode);
		rs.applyScaling(mat,mode);
	}
}

template <typename scalar, typename index>
std::array<bool,5> BSRMatrix<scalar,index,1>::isEqual(const BSRMatrix<scalar,index,1>& other,
                                                      const scalar tol) const
{
	return areEqual<scalar,index,1>(reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&mat),
	                                reinterpret_cast<const CRawBSRMatrix<scalar,index>*>(&other.mat),
	                                tol);
}

template <typename scalar, typename index>
index BSRMatrix<scalar,index,1>::zeroDiagonalRow() const
{
	int zdr = -1;
	for(index i = 0; i < mat.nbrows; i++)
		if(std::abs(mat.vals[mat.diagind[i]]) < 100*std::numeric_limits<scalar>::epsilon()) {
			zdr = i;
			break;
		}
	return zdr;
}

template <typename scalar, typename index>
size_t BSRMatrix<scalar,index,1>::getNumZeroDiagonals() const
{
	size_t num = 0;
	for(index i = 0; i < mat.nbrows; i++) {
		if(std::abs(mat.vals[mat.diagind[i]]) < 100*std::numeric_limits<scalar>::epsilon())
			num++;
	}
	return num;
}

template <typename scalar, typename index>
scalar BSRMatrix<scalar,index,1>::getDiagonalProduct() const
{
	scalar prod = 1;
	for(index i = 0; i < mat.nbrows; i++)
		prod *= mat.vals[mat.diagind[i]];
	return prod;
}

template <typename scalar, typename index>
scalar BSRMatrix<scalar,index,1>::getDiagonalAbsSum() const
{
	scalar sum = 0;
	for(index i = 0; i < mat.nbrows; i++)
		sum += std::abs(mat.vals[mat.diagind[i]]);
	return sum;
}

template <typename scalar, typename index>
scalar BSRMatrix<scalar,index,1>::getAbsMinDiagonalEntry() const
{
	scalar minentry = std::numeric_limits<scalar>::infinity();
	index minindex = 0;
	for(index i = 0; i < mat.nbrows; i++)
	{
		const scalar value = std::abs(mat.vals[mat.diagind[i]]);
		if(value < minentry) {
			minentry = value;
			minindex = i;
		}
	}
	std::cout << "Min entry = " << minentry << " at index " << minindex << std::endl;
	return minentry;
}

template <typename scalar, typename index>
scalar BSRMatrix<scalar,index,1>::getAbsMaxDiagonalEntry() const
{
	scalar maxentry = 0;
	for(index i = 0; i < mat.nbrows; i++)
	{
		const scalar value = std::abs(mat.vals[mat.diagind[i]]);
		if(value > maxentry)
			maxentry = value;
	}
	return maxentry;
}

}

