/** \file
 * \brief Implementations of some convenience functions for raw sparse-row matrix storage
 * \author Aditya Kashi
 */

#include <boost/align/aligned_alloc.hpp>
#include <srmatrixdefs.hpp>

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs>
RawBSRMatrix<scalar,index> copyRawBSRMatrix(const CRawBSRMatrix<scalar,index>& mat)
{
	RawBSRMatrix<scalar,index> nmat;
	nmat.nbrows = mat.nbrows;
	/*nmat.browptr = new index[mat.nbrows+1];
	nmat.bcolind = new index[mat.browptr[mat.nbrows]];
	nmat.vals = new scalar[mat.browptr[mat.nbrows]*bs*bs];
	nmat.diagind = new index[mat.nbrows];*/
	nmat.browptr = (index*)aligned_alloc(CACHE_LINE_LEN,(mat.nbrows+1)*sizeof(index));
	nmat.bcolind = (index*)aligned_alloc(CACHE_LINE_LEN,mat.browptr[mat.nbrows]*sizeof(index));
	nmat.vals = (scalar*)aligned_alloc(CACHE_LINE_LEN,mat.browptr[mat.nbrows]*bs*bs*sizeof(scalar));
	nmat.diagind = (index*)aligned_alloc(CACHE_LINE_LEN,mat.nbrows*sizeof(index));

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.browptr[mat.nbrows]; i++)
	{
		nmat.bcolind[i] = mat.bcolind[i];
		nmat.vals[i] = mat.vals[i];
	}

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows+1; i++)
		nmat.browptr[i] = mat.browptr[i];

#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.nbrows; i++)
		nmat.diagind[i] = mat.diagind[i];

	if(mat.browendptr)
		if(mat.nbrows > 0)
			nmat.browendptr = &nmat.browptr[1];

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
	lower.vals = mat.vals;

	index *lrowptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows+1)*sizeof(index));
	index *lrowendptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows)*sizeof(index));

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		lrowptr[irow] = mat.browptr[irow];
		lrowendptr[irow] = mat.diagind[irow]+1;
		for(index jj = mat.browptr[irow]; jj <= mat.diagind[irow]; jj++)
			nnz++;
	}
	lrowptr[mat.nbrows] = nnz;

	lower.browptr = lrowptr;
	lower.browendptr = lrowendptr;
	return lower;
}

template CRawBSRMatrix<double,int> getLowerTriangularView(const CRawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> getUpperTriangularView(const CRawBSRMatrix<scalar,index>& mat)
{
	CRawBSRMatrix<scalar,index> upper;
	upper.nbrows = mat.nbrows;
	upper.bcolind = mat.bcolind;
	upper.vals = mat.vals;

	index *urowptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows+1)*sizeof(index));
	index *urowendptr = (index*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(index));

	index nnz = 0;
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		urowptr[irow] = mat.diagind[irow];
		urowendptr[irow] = mat.browptr[irow+1];
		for(index jj = mat.diagind[irow]; jj < mat.browptr[irow+1]; jj++)
			nnz++;
	}
	urowptr[mat.nbrows] = nnz;

	upper.browptr = urowptr;
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
