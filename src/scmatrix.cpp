/** \file
 * \brief Implementation of some methods related to sparse-column storage of matrices
 * \author Aditya Kashi
 * \date 2018-08
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

#include <array>
#include <utility>
#include <vector>
#include <algorithm>
#include "scmatrixdefs.hpp"

namespace blasted {

template <typename scalar, typename index, int bs>
void convert_BSR_to_BSC(const CRawBSRMatrix<scalar,index> *const rmat,
                        CRawBSCMatrix<scalar,index> *const cscmat)
{
	constexpr int bs2 = bs*bs;

	RawBSCMatrix<scalar,index> cmat;
	const index bnnz = rmat->browptr[rmat->nbrows];
	const index N = rmat->nbrows;
	cmat.nbcols = N;
	cmat.bcolptr = (index*)aligned_alloc(CACHE_LINE_LEN, (N+1)*sizeof(index));
	cmat.browind = (index*)aligned_alloc(CACHE_LINE_LEN, bnnz*sizeof(index));
	cmat.vals = (scalar*)aligned_alloc(CACHE_LINE_LEN, bnnz*bs2*sizeof(scalar));
	cmat.diagind = (index*)aligned_alloc(CACHE_LINE_LEN, N*sizeof(index));

	// copy the values into a temporary column-wise storage

	using CSEntry = std::pair<index,std::array<scalar,bs2>>;
	std::vector<std::vector<CSEntry>> cv(N);
	const index expected_nnz_per_column = 30;
	for(auto it = cv.begin(); it != cv.end(); it++)
		it->reserve(expected_nnz_per_column);

	for(index irow = 0; irow < N; irow++) {
		for(index jj = rmat->browptr[irow]; jj < rmat->browptr[irow+1]; jj++)
		{
			std::array<scalar,bs2> bloc;
			for(int i = 0; i < bs2; i++)
				bloc[i] = rmat->vals[jj*bs2+i];
			cv[rmat->bcolind[jj]].push_back(std::make_pair(irow,bloc));
		}
	}

	// sort each of the columns

	//#pragma omp parallel for default(shared) schedule(dynamic, 100)
	for(index icol = 0; icol < N; icol++) {
		std::sort(cv[icol].begin(), cv[icol].end(),
		          [](CSEntry i, CSEntry j) { return i.first < j.first; });
	}

	// copy into output
	index iz = 0;
	for(index icol = 0; icol < N; icol++) {
		cmat.bcolptr[icol] = iz;
		cmat.diagind[icol] = -1;             // if there's no diagonal entry, we leave this as -1
		for(auto it = cv[icol].begin(); it != cv[icol].end(); it++) {
			cmat.browind[iz] = it->first;
			for(int i = 0; i < bs2; i++)
				cmat.vals[iz*bs2+i] = it->second[i];

			if(icol == cmat.browind[iz])
				cmat.diagind[icol] = iz;

			iz++;
		}
	}

	assert(iz == bnnz);
	cmat.bcolptr[N] = bnnz;

	cscmat->nbcols = cmat.nbcols;
	cscmat->bcolptr = cmat.bcolptr;
	cscmat->browind = cmat.browind;
	cscmat->vals = cmat.vals;
	cscmat->diagind = cmat.diagind;
}

template void
convert_BSR_to_BSC<double,int,1>(const CRawBSRMatrix<double,int> *const rmat,
                                 CRawBSCMatrix<double,int> *const cscmat);
template void
convert_BSR_to_BSC<double,int,3>(const CRawBSRMatrix<double,int> *const rmat,
                                 CRawBSCMatrix<double,int> *const cscmat);
template void
convert_BSR_to_BSC<double,int,4>(const CRawBSRMatrix<double,int> *const rmat,
                                 CRawBSCMatrix<double,int> *const cscmat);
template void
convert_BSR_to_BSC<double,int,5>(const CRawBSRMatrix<double,int> *const rmat,
                                 CRawBSCMatrix<double,int> *const cscmat);
template void
convert_BSR_to_BSC<double,int,7>(const CRawBSRMatrix<double,int> *const rmat,
                                 CRawBSCMatrix<double,int> *const cscmat);

template <typename scalar, typename index, int bs>
CRawBSCMatrix<scalar,index> convert_BSR_to_BSC_1based(const CRawBSRMatrix<scalar,index> *const rmat)
{
	RawBSCMatrix<scalar,index> cmat;
	convert_BSR_to_BSC<scalar,index,bs>(rmat, reinterpret_cast<CRawBSCMatrix<scalar,index>*>(&cmat));
	for(index i = 0; i < cmat.nbcols+1; i++)
		cmat.bcolptr[i] += 1;
	for(index i = 0; i < cmat.nbcols; i++)
		cmat.diagind[i] += 1;
	for(index i = 0; i < rmat->browptr[rmat->nbrows]; i++)
		cmat.browind[i] += 1;

	CRawBSCMatrix<scalar,index> ccmat;
	ccmat.nbcols =  cmat.nbcols;
	ccmat.bcolptr = cmat.bcolptr;
	ccmat.browind = cmat.browind;
	ccmat.diagind = cmat.diagind;
	ccmat.vals =    cmat.vals;
	return ccmat;
}

template CRawBSCMatrix<double,int>
convert_BSR_to_BSC_1based<double,int,1>(const CRawBSRMatrix<double,int> *const rmat);

template <typename scalar, typename index>
void alignedDestroyRawBSCMatrix(RawBSCMatrix<scalar,index>& cmat)
{
	aligned_free(cmat.bcolptr);
	aligned_free(cmat.browind);
	aligned_free(cmat.vals);
	aligned_free(cmat.diagind);
}

template <typename scalar, typename index>
void alignedDestroyCRawBSCMatrix(CRawBSCMatrix<scalar,index>& cmat)
{
	alignedDestroyRawBSCMatrix<scalar,index>(reinterpret_cast<RawBSCMatrix<scalar,index>&>(cmat));
}

template void alignedDestroyCRawBSCMatrix(CRawBSCMatrix<double,int>& cmat);

}
