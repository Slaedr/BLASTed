/** \file
 * \brief Implementations of some convenience functions for raw sparse-row matrix storage
 * \author Aditya Kashi
 */

#include <stdexcept>
#include <boost/align/aligned_alloc.hpp>
#include <srmatrixdefs.hpp>

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

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
void destroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat)
{
	delete [] rmat.browptr;
	delete [] rmat.bcolind;
	delete [] rmat.vals;
	delete [] rmat.diagind;
	if(rmat.nnzb != rmat.nbstored)
		delete [] rmat.browendptr;
}

template void destroyCRawBSRMatrix(CRawBSRMatrix<double,int>& rmat);

template <typename scalar, typename index>
void destroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat) {
	destroyCRawBSRMatrix(reinterpret_cast<CRawBSRMatrix<scalar,index>&>(rmat));
}

template void destroyRawBSRMatrix(RawBSRMatrix<double,int>& rmat);

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
void alignedDestroyRawBSRMatrixTriangularView(RawBSRMatrix<scalar,index>& mat)
{
	aligned_free(mat.browptr);
	aligned_free(mat.browendptr);
}

template void alignedDestroyRawBSRMatrixTriangularView(RawBSRMatrix<double,int>& mat);

}
