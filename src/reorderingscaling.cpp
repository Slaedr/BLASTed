/** \file
 * \brief Implementation of schemes to reorder and scale matrices for various purposes
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

#include <algorithm>
#include <utility>
#include "helper_algorithms.hpp"
#include "reorderingscaling.hpp"
#include "scmatrixdefs.hpp"

namespace blasted {

/** For the output invp and input p, it can be shown that
 * invp[p[i]] = i  (obviously, see below), and also
 * p[invp[i]] = i.
 */
template <typename index>
std::vector<index> invertPermutationVector(const std::vector<index> p)
{
	std::vector<index> invp(p.size());
	for(size_t i = 0; i < p.size(); i++)
		invp[p[i]] = i;
	return invp;
}

template <typename scalar, typename index, int bs>
Reordering<scalar,index,bs>::Reordering()
{ }

template <typename scalar, typename index, int bs>
Reordering<scalar,index,bs>::~Reordering()
{ }

template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::setOrdering(const index *const rord, const index *const cord,
                                              const index length)
{
	if(rord) {
		rp.resize(length);
		for(index i = 0; i < length; i++)
			rp[i] = rord[i];

		// compute inverse ordering
		irp.resize(length);
		for(index i = 0; i < length; i++)
			irp[rord[i]] = i;
	}

	if(cord) {
		cp.resize(length);
		for(index i = 0; i < length; i++)
			cp[i] = cord[i];

		icp.resize(length);
		for(index i = 0; i < length; i++)
			icp[cord[i]] = i;
	}
}

template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::applyOrdering(RawBSRMatrix<scalar,index>& mat,
                                                const RSApplyMode mode) const
{
	if(mode == FORWARD) {
		if(rp.size() > 0)
		{
			// move rows around
			std::vector<scalar> tempval(mat.browptr[mat.nbrows]*bs*bs);
			std::vector<index> tempcind(mat.browptr[mat.nbrows]);
			std::vector<index> temprptr(mat.nbrows+1);
			temprptr[mat.nbrows] = mat.browptr[mat.nbrows];

			index pos = 0;

			for(index i = 0; i < mat.nbrows; i++)
			{
				const index ni = rp[i];
				temprptr[i] = pos;
				for(index jj = mat.browptr[ni]; jj < mat.browptr[ni+1]; jj++) {
					tempcind[pos] = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						tempval[pos*bs*bs + k] = mat.vals[jj*bs*bs+k];
					pos++;
				}
			}

			assert(pos == mat.browptr[mat.nbrows]);

			// copy into original array
			for(index i = 0; i < mat.nbrows; i++)
				mat.browptr[i] = temprptr[i];
			for(index jj = 0; jj < pos; jj++) {
				mat.bcolind[jj] = tempcind[jj];
				for(int k = 0; k < bs*bs; k++)
					mat.vals[jj*bs*bs+k] = tempval[jj*bs*bs + k];
			}
		}

		if(cp.size() > 0)
		{
#pragma omp parallel for default(shared) schedule(dynamic,200)
			for(index i = 0; i < mat.nbrows; i++)
			{
				index *const rcolind = &mat.bcolind[mat.browptr[i]];
				scalar *const rvals = &mat.vals[mat.browptr[i]*bs*bs];
				const index rsize = mat.browptr[i+1]-mat.browptr[i];

				// copy column indices of this row into a temp vector
				const std::vector<index> ocinds(rcolind, rcolind+rsize);

				// transform column indices with the inverse permutation, so that
				//  the non-zero values transformed with the forward permutation.
				for(index jj = 0; jj < rsize; jj++)
					rcolind[jj] = icp[ocinds[jj]];

				internal::sortBlockInnerDimension<scalar,index,bs>(rsize, rcolind, rvals);
			}
		}
	}
	else {
		// inverse ordering

		if(rp.size() > 0)
		{
			// copy the matrix to a list of rows
			using CSEntry = std::pair<index,scalar[bs*bs]>;
			std::vector<std::vector<CSEntry>> brows(mat.nbrows);

			// can be parallelized
			for(index i = 0; i < mat.nbrows; i++)
			{
				// Row rp[i] in the permuted matrix gets the i-th row of the original matrix
				const index destr = rp[i];
				assert(brows[destr].size() == 0);
				brows[destr].resize(mat.browptr[i+1]-mat.browptr[i]);

				for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				{
					const index dj = jj-mat.browptr[i];
					brows[destr][dj].first = mat.bcolind[jj];
					for(int k = 0; k < bs*bs; k++)
						brows[destr][dj].second[k] = mat.vals[jj*bs*bs+k];
				}
			}

			// copy back - NOT parallel
			mat.browptr[0] = 0;
			for(index i = 0; i < mat.nbrows; i++)
			{
				mat.browptr[i+1] = static_cast<index>(brows[i].size()) + mat.browptr[i];
				for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				{
					const index dj = jj-mat.browptr[i];
					mat.bcolind[jj] = brows[i][dj].first;
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs+k] = brows[i][dj].second[k];
				}
			}
		}

		if(cp.size() > 0)
		{
			// rename columns
#pragma omp parallel for default(shared) schedule(dynamic,200)
			for(index i = 0; i < mat.nbrows; i++)
			{
				index *const rcolind = &mat.bcolind[mat.browptr[i]];
				scalar *const rvals = &mat.vals[mat.browptr[i]*bs*bs];
				const index rsize = mat.browptr[i+1]-mat.browptr[i];

				// copy column indices of this row into a temp vector
				const std::vector<index> ocinds(rcolind, rcolind+rsize);

				// transform column indices with the forward permutation, so that
				//  the actual matrix is transformed with the inverse permutation.
				for(index jj = 0; jj < rsize; jj++)
					rcolind[jj] = cp[ocinds[jj]];

				internal::sortBlockInnerDimension<scalar,index,bs>(rsize, rcolind, rvals);
			}
		}
	}
}

/** This is most likely not the best way to do it. The vector is first copied into a local
 * temporary storage which is freed in the end.
 */
template <typename scalar, typename index, int bs>
void Reordering<scalar,index,bs>::applyOrdering(scalar *const vec,
                                                const RSApplyMode mode, const RSApplyDir dir) const
{
	const index size = dir == ROW ? rp.size() : cp.size();
	if (size <= 0) {
		return;
	}

	// copy vector to temp location
	std::vector<scalar> tv(size*bs);
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < size*bs; i++)
		tv[i] = vec[i];

	if(mode == FORWARD)
	{
		if(dir == ROW)
		{
			// apply row ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[i*bs+k] = tv[rp[i]*bs+k];
		}
		else {
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[i*bs+k] = tv[cp[i]*bs+k];
		}
	}
	else {
		if(dir == ROW)
		{
			// apply inverse row ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[rp[i]*bs+k] = tv[i*bs+k];
		}
		else {
			// apply inverse column ordering
#pragma omp parallel for default(shared)
			for(index i = 0; i < size; i++)
				for(int k = 0; k < bs; k++)
					vec[cp[i]*bs+k] = tv[i*bs+k];
		}
	}
}

template class Reordering<double,int,1>;
template class Reordering<double,int,4>;
template class Reordering<double,int,7>;

template <typename scalar, typename index, int bs>
ReorderingScaling<scalar,index,bs>::ReorderingScaling()
	: Reordering<scalar,index,bs>()
{ }

template <typename scalar, typename index, int bs>
ReorderingScaling<scalar,index,bs>::~ReorderingScaling()
{ }

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index,bs>::applyScaling(RawBSRMatrix<scalar,index>& mat,
                                                      const RSApplyMode mode) const
{
	if(rowscale.size() > 0)
	{
		// scale rows
		if(mode == FORWARD) {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				const scalar rowscaler = rowscale[irow];
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] *= rowscaler;
			}
		}
		else {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				const scalar rowscaler = rowscale[irow];
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] /= rowscaler;
			}
		}
	}

	if(colscale.size() > 0)
	{
		// scale columns
		if(mode == FORWARD) {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++) {
					const scalar colscaler = colscale[mat.bcolind[jj]];
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] *= colscaler;
				}
			}
		}
		else {
#pragma omp parallel for default(shared) schedule(dynamic, 200)
			for(index irow = 0; irow < mat.nbrows; irow++)
			{
				for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++) {
					const scalar colscaler = colscale[mat.bcolind[jj]];
					for(int k = 0; k < bs*bs; k++)
						mat.vals[jj*bs*bs + k] /= colscaler;
				}
			}
		}
	}
}

template <typename scalar, typename index, int bs>
void ReorderingScaling<scalar,index,bs>::applyScaling(scalar *const vec, const RSApplyMode mode,
                                                      const RSApplyDir dir) const
{
	if(mode == FORWARD)
	{
		if(dir == ROW)
		{
			if(rowscale.size() > 0)
				// apply row scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(rowscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] *= rowscale[i];
		}
		else
		{
			if(colscale.size() > 0)
				// apply column scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(colscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] *= colscale[i];
		}
	}
	else {
		if(dir == ROW)
		{
			if(rowscale.size() > 0)
				// apply inverse row scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(rowscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] /= rowscale[i];
		}
		else {
			if(colscale.size() > 0)
				// apply inverse column scaling
#pragma omp parallel for default(shared)
				for(index i = 0; i < static_cast<index>(colscale.size()); i++)
					for(int k = 0; k < bs; k++)
						vec[i*bs+k] /= colscale[i];
		}
	}
}

template class ReorderingScaling<double,int,1>;
template class ReorderingScaling<double,int,4>;
template class ReorderingScaling<double,int,7>;

}

#ifdef HAVE_MC64

extern "C" {

	extern void mc64a_(const int *const job, const int *const n, const int *const ne,
	                   const int *const colptr, const int *const rowind, const float *const A,
	                   int *const num_diag, int *const cperm,
	                   const int *const len_workvec, int *const workvec,
	                   const int *const len_scalevec, float *const scalevec,
	                   int icntl[10], int info[10]);

	extern void mc64ad_(const int *const job, const int *const n, const int *const ne,
	                    const int *const colptr, const int *const rowind, const double *const A,
	                    int *const num_diag, int *const cperm,
	                    const int *const len_workvec, int *const workvec,
	                    const int *const len_scalevec, double *const scalevec,
	                    int icntl[10], int info[10]);

	extern void mc64id_(int icntl[10]);
}

namespace blasted {

MC64::MC64(const int jobid)
	: job{jobid}
{ }


// Returns the size of the two work vectors required by MC64
/* \return Refer to the MC64 documentation.
 * The first entry is the size LIW of work vector IW.
 * The second entry is the size LDW of the work vector DW.
 */
std::tuple<int,int> getWorkSizes(const int job, const int nrows, const int nnz)
{
	int liw = 0, ldw = 0;
	switch(job) {
	case 1:
		liw = 5*nrows;
		ldw = 0;
		break;
	case 2:
		liw = 4*nrows;
		ldw = nrows;
		break;
	case 3:
		liw = 10*nrows + nnz;
		ldw = nnz;
		break;
	case 4:
		liw = 5*nrows;
		ldw = 2*nrows + nnz;
		break;
	case 5:
		liw = 5*nrows;
		ldw = 3*nrows + nnz;
		break;
	default:
		throw std::runtime_error("Invalid MC64 job!");
	}

	return std::make_tuple(liw, ldw);
}

void MC64::compute(const CRawBSRMatrix<double,int>& mat)
{
	const CRawBSCMatrix<double,int> scmat = convert_BSR_to_BSC_1based<double,int,1>(&mat);
	assert(mat.nbrows == scmat.nbcols);
	assert(mat.browptr[mat.nbrows]+1 == scmat.bcolptr[scmat.nbcols]);
	const int nnz = mat.browptr[mat.nbrows];

	cp.resize(mat.nbrows);
	if(job == 5) {
		rowscale.resize(mat.nbrows);
		colscale.resize(mat.nbrows);
	}

	const std::tuple<int,int> worklen = getWorkSizes(job, mat.nbrows, nnz);
	std::vector<int> workvec(std::get<0>(worklen));
	std::vector<double> scalevec(std::get<1>(worklen));

	int icntl[10];
	mc64id_(icntl);

	// Set options in icntl if needed.
	//icntl[3] = 1;                        //  disables data-checking.

	int num_diag, info[10];

	mc64ad_(&job, &scmat.nbcols, &nnz, scmat.bcolptr, scmat.browind, scmat.vals,
	        &num_diag, &cp[0], &std::get<0>(worklen), &workvec[0],
	        &std::get<1>(worklen), &scalevec[0], icntl, info);

	// Check status flags in info
	if(info[0] != 0) {
		if(info[0] == 1)
			throw std::runtime_error("MC64: Matrix was structurally singular!");
		else
			throw std::runtime_error("MC64 failed! Error code = " + std::to_string(info[0]));
	}

	if(num_diag < scmat.nbcols)
	{
		printf("! MC64: Permuted matrix has %d nonzeros on its diagonal!\n", num_diag);
		fflush(stdout);
	}

	destroyRawBSCMatrix<double,int>(scmat);

	const double *const colscalevec = (job == 5) ? &scalevec[mat.nbrows] : nullptr;

	for(int i = 0; i < mat.nbrows; i++) {
		if(job == 5) {
			rowscale[i] = std::exp(scalevec[i]);
			colscale[i] = std::exp(colscalevec[i]);
		}
		// MC64 requires taking abs value. Also convert to 0-based indexing.
		cp[i] = std::abs(cp[i])-1;
	}

	// compute the inverse column permutation as well
	icp = invertPermutationVector<int>(cp);
}

}

#endif

