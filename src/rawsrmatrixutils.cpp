/** \file
 * \brief Implementations of some convenience functions for raw sparse-row matrix storage
 * \author Aditya Kashi
 */

#include <srmatrixdefs.hpp>

namespace blasted {

template <typename scalar, typename index, int bs>
RawBSRMatrix<scalar,index> copyRawBSRMatrix(const CRawBSRMatrix<scalar,index>& mat)
{
	RawBSRMatrix<scalar,index> nmat;
	nmat.nbrows = mat.nbrows;
	nmat.browptr = new index[mat.nbrows+1];
	nmat.bcolind = new index[mat.browptr[mat.nbrows]];
	nmat.vals = new scalar[mat.browptr[mat.nbrows]*bs*bs];
	nmat.diagind = new index[mat.nbrows];

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

	return nmat;
}

}
