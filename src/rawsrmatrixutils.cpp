/** \file
 * \brief Implementations of some convenience functions for raw sparse-row matrix storage
 * \author Aditya Kashi
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

#include <stdexcept>
#include <boost/align/aligned_alloc.hpp>
#include <srmatrixdefs.hpp>

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename mscalar, typename mindex>
SRMatrixStorage<mscalar,mindex>::SRMatrixStorage() : nbrows{0}, nnzb{0}, nbstored{0}
{ }

template <typename mscalar, typename mindex>
SRMatrixStorage<mscalar,mindex>::SRMatrixStorage(mindex *const brptrs, mindex *const bcinds,
                                                 mscalar *const values, mindex *const diag_inds,
                                                 mindex *const brendptrs,
                                                 const index n_brows, const index n_nzb,
                                                 const index n_bstored, const int b_s)
	: browptr(brptrs, n_brows+1), bcolind(bcinds, n_bstored), vals(values, n_bstored*b_s*b_s),
	  diagind(diag_inds, n_brows), browendptr(brendptrs, n_brows),
	  nbrows{n_brows}, nnzb{n_nzb}, nbstored{n_bstored}
{ }

template <typename mscalar, typename mindex>
SRMatrixStorage<mscalar,mindex>::SRMatrixStorage(ArrayView<mindex>&& brptrs, ArrayView<mindex>&& bcinds,
                                                 ArrayView<mscalar>&& values,
                                                 ArrayView<mindex>&& diaginds,
                                                 ArrayView<mindex>&& brendptrs,
                                                 const index n_brows, const index n_nzb,
                                                 const index n_bstored)
	: browptr(std::move(brptrs)), bcolind(std::move(bcinds)), vals(std::move(values)),
	  diagind(std::move(diaginds)), browendptr(std::move(brendptrs)),
	  nbrows{n_brows}, nnzb{n_nzb}, nbstored{n_bstored}
{ }

template <typename mscalar, typename mindex>
SRMatrixStorage<mscalar,mindex>::SRMatrixStorage(SRMatrixStorage<mscalar,mindex>&& other)
	: browptr(std::move(other.browptr)), bcolind(std::move(other.bcolind)), vals(std::move(other.vals)),
	  diagind(std::move(other.diagind)), browendptr(std::move(other.browendptr)),
	  nbrows{other.nbrows}, nnzb{other.nnzb}, nbstored{other.nbstored}
{
}

template <typename scalar, typename index>
SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
move_to_const(SRMatrixStorage<scalar,index>&& smat)
{
	SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
		cm(std::move(move_to_const(std::move(smat.browptr))),
		   std::move(move_to_const(std::move(smat.bcolind))),
		   std::move(move_to_const(std::move(smat.vals))),
		   std::move(move_to_const(std::move(smat.diagind))),
		   std::move(move_to_const(std::move(smat.browendptr))),
		   smat.nbrows, smat.nnzb, smat.nbstored
		   );

	smat.nbrows = smat.nnzb = smat.nbstored = 0;
	return cm;
}

template struct SRMatrixStorage<double,int>;
template struct SRMatrixStorage<const double,const int>;
template SRMatrixStorage<typename std::add_const<double>::type, typename std::add_const<int>::type>
move_to_const(SRMatrixStorage<double,int>&& smat);

template <typename scalar, typename index>
SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
share_with_const(const SRMatrixStorage<scalar,index>& smat, const int bs)
{
	return
		SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
		(&smat.browptr[0], &smat.bcolind[0], &smat.vals[0], &smat.diagind[0], &smat.browendptr[0],
		 smat.nbrows, smat.nnzb, smat.nbstored, bs);
}

template SRMatrixStorage<typename std::add_const<double>::type, typename std::add_const<int>::type>
share_with_const(const SRMatrixStorage<double,int>& smat, const int bs);

// Makes a shallow copy of a matrix
/* Copies over pointers to the underlying storage to create new ArrayViews and uses them to create
 * a SRMatrixStorage. The arrays DO NOT take ownership of the underlying storage, which is not
 * de-allocated when the copied matrix is destroyed.
 */
// template <typename scalar, typename index>
// SRMatrixStorage<scalar,index> shallow_copy(const SRMatrixStorage<scalar,index>& mat)
// {
// 	ArrayView<index> brptrs(&mat.browptr[0], mat.browptr.size());
// 	ArrayView<index> bcinds(&mat.bcolind[0], mat.bcolind.size());
// 	ArrayView<scalar> values(&mat.vals[0], mat.vals.size());
// 	ArrayView<index> brendptrs(&mat.browendptr[0], mat.browendptr.size());
// 	ArrayView<index> dinds(&mat.diagind[0], mat.diagind.size());

// 	return SRMatrixStorage<scalar,index>(std::move(brptrs), std::move(bcinds), std::move(values),
// 	                                     std::move(dinds), std::move(brendptrs), mat.nbrows,
// 	                                     mat.nnzb, mat.nbstored);
// }

// template SRMatrixStorage<double,int> shallow_copy(const SRMatrixStorage<double,int>&);

template <typename scalar, typename index, int bs>
RawBSRMatrix<scalar,index> copyRawBSRMatrix(const CRawBSRMatrix<scalar,index>& mat)
{
	constexpr int bs2 = bs*bs;

	RawBSRMatrix<scalar,index> nmat;
	nmat.nbrows = mat.nbrows;
	nmat.browptr = (index*)aligned_alloc(CACHE_LINE_LEN,(mat.nbrows+1)*sizeof(index));
	nmat.bcolind = (index*)aligned_alloc(CACHE_LINE_LEN,mat.browptr[mat.nbrows]*sizeof(index));
	nmat.vals = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.browptr[mat.nbrows]*bs*bs*sizeof(scalar));
	nmat.diagind = (index*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(index));
	nmat.nnzb = mat.nnzb;
	nmat.nbstored = mat.nbstored;

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbstored; i++)
	{
		nmat.bcolind[i] = mat.bcolind[i];
		for(int j = 0; j < bs2; j++)
			nmat.vals[i*bs2+j] = mat.vals[i*bs2+j];
	}

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows+1; i++)
		nmat.browptr[i] = mat.browptr[i];

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows; i++)
		nmat.diagind[i] = mat.diagind[i];

	if(mat.browendptr) {
		if(mat.nbrows > 0)
			nmat.browendptr = &nmat.browptr[1];
	}
	else {
		throw std::runtime_error("CRawBSRMatrix does not have row end pointers!");
	}

	return nmat;
}

template RawBSRMatrix<double,int> copyRawBSRMatrix<double,int,1>(const CRawBSRMatrix<double,int>& mat);
template RawBSRMatrix<double,int> copyRawBSRMatrix<double,int,4>(const CRawBSRMatrix<double,int>& mat);
template RawBSRMatrix<double,int> copyRawBSRMatrix<double,int,5>(const CRawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
void alignedDestroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat)
{
	aligned_free(rmat.browptr);
	aligned_free(rmat.bcolind);
	aligned_free(rmat.diagind);
	aligned_free(rmat.vals);
	if(rmat.nnzb != rmat.nbstored)
		aligned_free(rmat.browendptr);
}

template void alignedDestroyRawBSRMatrix(RawBSRMatrix<double,int>& rmat);

template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> getLowerTriangularView(const CRawBSRMatrix<scalar,index>& mat)
{
	CRawBSRMatrix<scalar,index> lower;
	lower.nbrows = mat.nbrows;
	lower.bcolind = mat.bcolind;
	lower.diagind = mat.diagind;
	lower.vals = mat.vals;
	lower.nbstored = mat.nbstored;

	index *lrowptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows)*sizeof(index));
	index *lrowendptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows)*sizeof(index));

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		lrowptr[irow] = mat.browptr[irow];
		lrowendptr[irow] = mat.diagind[irow]+1;
		nnz += (mat.diagind[irow]-mat.browptr[irow]+1);
	}
	//lrowptr[mat.nbrows] = nnz;
	lower.nnzb = nnz;

	lower.browptr = lrowptr;
	lower.browendptr = lrowendptr;
	return lower;
}

template CRawBSRMatrix<double,int> getLowerTriangularView(const CRawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> getUpperTriangularView(const CRawBSRMatrix<scalar,index>& mat)
{
	assert(mat.browendptr[mat.nbrows-1] == mat.diagind[mat.nbrows-1]+1);

	CRawBSRMatrix<scalar,index> upper;
	upper.nbrows = mat.nbrows;
	upper.bcolind = mat.bcolind;
	upper.diagind = mat.diagind;
	upper.vals = mat.vals;
	upper.nbstored = mat.nbstored;

	index *urowptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows)*sizeof(index));
	index *urowendptr = (index*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(index));

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		urowptr[irow] = mat.diagind[irow];
		urowendptr[irow] = mat.browendptr[irow];
		nnz += (mat.browendptr[irow]-mat.diagind[irow]);
	}
	//urowptr[mat.nbrows] = nnz;
	upper.nnzb = nnz;

	upper.browptr = urowptr;
	upper.browendptr = urowendptr;

	assert(upper.browendptr[upper.nbrows-1] == upper.diagind[upper.nbrows-1]+1);

#ifdef DEBUG
	for(index irow = 0; irow < upper.nbrows; irow++)
	{
		for(index jj = upper.browptr[irow]; jj < upper.browendptr[irow]; jj++)
			assert(upper.bcolind[jj] > irow || upper.diagind[irow] == jj);
	}
#endif
	return upper;
}

template CRawBSRMatrix<double,int> getUpperTriangularView(const CRawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
SRMatrixStorage<const scalar,const index>
getLowerTriangularView(const SRMatrixStorage<const scalar,const index>&& mat)
{
	ArrayView<const index> lbcolind(&mat.bcolind[0], mat.bcolind.size());
	ArrayView<const index> ldiagind(&mat.diagind[0], mat.diagind.size());
	ArrayView<const scalar> lvals(&mat.vals[0], mat.vals.size());

	ArrayView<index> lrowptr(mat.nbrows);
	ArrayView<index> lrowendptr(mat.nbrows);

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		lrowptr[irow] = mat.browptr[irow];
		lrowendptr[irow] = mat.diagind[irow]+1;
		nnz += (mat.diagind[irow]-mat.browptr[irow]+1);
	}

	SRMatrixStorage<const scalar,const index>
		lower(std::move(move_to_const(std::move(lrowptr))), std::move(lbcolind), std::move(lvals),
		      std::move(ldiagind), std::move(move_to_const(std::move(lrowendptr))),
		      mat.nbrows, nnz, mat.nbstored);

	return lower;
}

template SRMatrixStorage<const double,const int>
getLowerTriangularView(const SRMatrixStorage<const double,const int>&& mat);

template <typename scalar, typename index>
SRMatrixStorage<const scalar,const index>
getUpperTriangularView(const SRMatrixStorage<const scalar,const index>&& mat)
{
	assert(mat.browendptr[mat.nbrows-1] == mat.diagind[mat.nbrows-1]+1);

	ArrayView<const index> ubcolind(&mat.bcolind[0], mat.bcolind.size());
	ArrayView<const index> udiagind(&mat.diagind[0], mat.diagind.size());
	ArrayView<const scalar> uvals(&mat.vals[0], mat.vals.size());

	ArrayView<index> urowptr(mat.nbrows);
	ArrayView<index> urowendptr(mat.nbrows);

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		urowptr[irow] = mat.diagind[irow];
		urowendptr[irow] = mat.browendptr[irow];
		nnz += (mat.browendptr[irow]-mat.diagind[irow]);
	}

	SRMatrixStorage<const scalar,const index>
		upper(std::move(move_to_const(std::move(urowptr))), std::move(ubcolind), std::move(uvals),
		      std::move(udiagind), std::move(move_to_const(std::move(urowendptr))),
		      mat.nbrows, nnz, mat.nbstored);

#ifdef DEBUG
	assert(upper.browendptr[upper.nbrows-1] == upper.diagind[upper.nbrows-1]+1);

	for(index irow = 0; irow < upper.nbrows; irow++)
	{
		for(index jj = upper.browptr[irow]; jj < upper.browendptr[irow]; jj++)
			assert(upper.bcolind[jj] > irow || upper.diagind[irow] == jj);
	}
#endif
	return upper;
}

template SRMatrixStorage<const double,const int>
getUpperTriangularView(const SRMatrixStorage<const double,const int>&& mat);

template <typename scalar, typename index>
void alignedDestroyRawBSRMatrixTriangularView(RawBSRMatrix<scalar,index>& mat)
{
	aligned_free(mat.browptr);
	aligned_free(mat.browendptr);
}

template void alignedDestroyRawBSRMatrixTriangularView(RawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> createRawView(const SRMatrixStorage<const scalar, const index>&& smat)
{
	CRawBSRMatrix<scalar,index> cmat;
	cmat.nbrows = smat.nbrows; cmat.nnzb = smat.nnzb; cmat.nbstored = smat.nbstored;
	cmat.browptr = &smat.browptr[0]; cmat.browendptr = &smat.browendptr[0];
	cmat.bcolind = &smat.bcolind[0]; cmat.diagind = &smat.diagind[0]; cmat.vals = &smat.vals[0];

	return cmat;
}

template CRawBSRMatrix<double,int> createRawView(const SRMatrixStorage<const double, const int>&& smat);

template <typename scalar, typename index, int bs>
void getScalingVector(const CRawBSRMatrix<scalar,index> *const mat, scalar *const __restrict scale)
{
#pragma omp parallel for simd default(shared)
	for(int i = 0; i < mat->nbrows; i++)
		for(int j = 0; j < bs; j++)
			scale[i*bs + j] = 1.0/std::sqrt(mat->vals[mat->diagind[i]*bs*bs + j*bs + j]);
}

template void getScalingVector<double,int,1>(const CRawBSRMatrix<double,int> *const mat,
                                             double *const __restrict scale);
template void getScalingVector<double,int,4>(const CRawBSRMatrix<double,int> *const mat,
                                             double *const __restrict scale);
template void getScalingVector<double,int,5>(const CRawBSRMatrix<double,int> *const mat,
                                             double *const __restrict scale);
template void getScalingVector<double,int,7>(const CRawBSRMatrix<double,int> *const mat,
                                             double *const __restrict scale);

}
