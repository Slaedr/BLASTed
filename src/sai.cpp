/** \file
 * \brief Implementation of some functionality useful for building SAI-type preconditioners
 */

#include <set>
#include <Eigen/Dense>
#include "sai.hpp"
#include "helper_algorithms.hpp"

namespace blasted {

template <typename scalar, typename index>
LeftSAIPattern<index> left_SAI_pattern(const SRMatrixStorage<const scalar,const index>&& mat)
{
	LeftSAIPattern<index> tsp;

	/* Note that every block-row of A corresponds to a block-row of the approx inverse M and to a
	 * least-squares problem.
	 */

	tsp.sairowptr.assign(mat.nbrows+1,0);
	tsp.nVars.assign(mat.nbrows,0);
	tsp.nEqns.assign(mat.nbrows,0);
	tsp.localCentralRow.assign(mat.nbrows, -1);

	// Step 1: Compute number of variables and constraints for each least-squares problem

	index totalcoeffs = 0;     // number of least-square LHS coeffs over all block-rows
	//#pragma omp parallel for default(shared) reduction(+:totalcoeffs)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		tsp.nVars[irow] = mat.browendptr[irow] - mat.browptr[irow];

		std::set<index> constraints;
		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A

			for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++) {
				constraints.insert(mat.bcolind[kk]);
				totalcoeffs++;
			}
		}

		tsp.nEqns[irow] = static_cast<int>(constraints.size());
		tsp.sairowptr[irow+1] = tsp.nVars[irow];
	}

	// get the starting of each least-squares problem in bcolptr
	internal::inclusive_scan(tsp.sairowptr);

	const index totalvars = tsp.sairowptr[mat.nbrows];
#ifdef DEBUG
	for(index irow = 0; irow < mat.nbrows; irow++)
		assert(tsp.sairowptr[irow] < totalvars);
#endif
	tsp.bcolptr.assign(totalvars+1,0);
	tsp.bpos.assign(totalcoeffs,0);
	tsp.browind.assign(totalcoeffs,0);

	// Step 2: Get pointers into bpos and browind corresponding to the beginning of
	//  each column in each least-squares matrix over all rows of the orig matrix

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			const index localcolidx = jj - mat.browptr[irow];

			for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
			{
				assert(tsp.sairowptr[irow]+localcolidx < tsp.sairowptr[irow+1]);
				tsp.bcolptr[tsp.sairowptr[irow] + localcolidx + 1]++;
			}
		}
	}

	internal::inclusive_scan(tsp.bcolptr);
	assert(tsp.bcolptr[tsp.sairowptr[mat.nbrows]] == totalcoeffs);

	// Step 3: For the least-squares problem of each row of A, compute the sparsity pattern

	// #pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		// Get local row indices (in LHS matrix, which is a block of A^T)

		std::set<index> constraints;
		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A

			for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
				constraints.insert(mat.bcolind[kk]);
		}

		const std::vector<index> constrvec(constraints.begin(), constraints.end());
		assert(static_cast<int>(constrvec.size())==tsp.nEqns[irow]);

		// find constraint corresponding to the diagonal entry irow
		for(int icons = 0; icons < tsp.nEqns[irow]; icons++)
			if(irow == constrvec[icons])
				tsp.localCentralRow[irow] = icons;

		assert(tsp.localCentralRow[irow] != -1);

		std::vector<std::vector<int>> localrowinds(tsp.nVars[irow]);

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];
			const int localcolidx = jj - mat.browptr[irow];
			localrowinds[localcolidx].assign(mat.browendptr[col]-mat.browptr[col], -1);

			for(size_t i = 0; i < constrvec.size(); i++)
			{
				for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
					if(constrvec[i] == mat.bcolind[kk])
					{
						localrowinds[localcolidx][kk-mat.browptr[col]] = i;
					}
			}
		}

#ifdef DEBUG
		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const int localcolidx = jj - mat.browptr[irow];
			for(size_t i = 0; i < localrowinds[localcolidx].size(); i++) {
				assert(localrowinds[localcolidx][i] >= 0);
				assert(localrowinds[localcolidx][i] < tsp.nEqns[irow]);
			}
		}
#endif

		/* Note that because sets are always ordered, the set 'constraints' is ordered by column index.
		 * This means that localrowinds is ordered by column index of the constraints.
		 */

		// Then, store positions and local row indices

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			const int localcolidx = jj - mat.browptr[irow];

			for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
			{
				const int locpos = kk - mat.browptr[col];
				tsp.bpos[tsp.bcolptr[tsp.sairowptr[irow]+localcolidx] + locpos] = kk;
				tsp.browind[tsp.bcolptr[tsp.sairowptr[irow]+localcolidx] + locpos]
					= localrowinds[localcolidx][locpos];
			}
		}

		// sanity check
#ifdef DEBUG
		for(index icol = tsp.sairowptr[irow]; icol < tsp.sairowptr[irow+1]; icol++)
		{
			for(index j = tsp.bcolptr[icol]; j < tsp.bcolptr[icol+1]; j++)
			{
				assert(tsp.bpos[j] < mat.nbstored);
				assert(tsp.browind[j] >= 0);
				assert(tsp.browind[j] < tsp.nEqns[irow]);
			}
		}
#endif
	}

	return tsp;
}

template LeftSAIPattern<int> left_SAI_pattern(const SRMatrixStorage<const double,const int>&& mat);

template <typename scalar, typename index>
LeftSAIPattern<index> left_incomplete_SAI_pattern(const SRMatrixStorage<const scalar,const index>&& mat)
{
	LeftSAIPattern<index> tsp;

	/* Note that every block-row of A corresponds to a block-row of the approx inverse M and to a
	 * least-squares problem.
	 */

	tsp.sairowptr.assign(mat.nbrows+1,0);
	tsp.nVars.resize(mat.nbrows);
	tsp.nEqns.resize(mat.nbrows);
	tsp.localCentralRow.assign(mat.nbrows, -1);

	// Compute sizes

	index totalcoeffs = 0;

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		tsp.nVars[irow] = mat.browendptr[irow] - mat.browptr[irow];
		tsp.nEqns[irow] = tsp.nVars[irow];
		tsp.sairowptr[irow+1] = tsp.nVars[irow];

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index colind = mat.bcolind[jj];
			for(index kk = mat.browptr[colind]; kk < mat.browendptr[colind]; kk++)
			{
				for(index ll = mat.browptr[irow]; ll < mat.browendptr[irow]; ll++) {
					if(mat.bcolind[ll] == mat.bcolind[kk]) {
						totalcoeffs++;
					}
				}
			}
		}
	}

	internal::inclusive_scan(tsp.sairowptr);

	const index totalvars = tsp.sairowptr[mat.nbrows];
	tsp.bcolptr.resize(totalvars+1);
	tsp.browind.resize(totalcoeffs);
	tsp.bpos.resize(totalcoeffs);

	// Compute pointers to the beginning of every column in the LHS matrix for every row of the SAI
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		const index startcol = tsp.sairowptr[irow];

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index loccol = jj - mat.browptr[irow];
			const index colind = mat.bcolind[jj];
			for(index kk = mat.browptr[colind]; kk < mat.browendptr[colind]; kk++)
			{
				for(index ll = mat.browptr[irow]; ll < mat.browendptr[irow]; ll++) {
					if(mat.bcolind[ll] == mat.bcolind[kk])
					{
						assert(startcol+loccol < tsp.sairowptr[irow+1]);
						tsp.bcolptr[startcol+loccol+1]++;
					}
				}
			}

			if(colind == irow)
				tsp.localCentralRow[irow] = loccol;
		}

		assert(tsp.localCentralRow[irow] >= 0);
		assert(tsp.localCentralRow[irow] < tsp.nVars[irow]);
	}

	internal::inclusive_scan(tsp.bcolptr);
	assert(tsp.bcolptr[totalvars]==totalcoeffs);

	// Compute the pattern

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		const index startcol = tsp.sairowptr[irow];

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index colind = mat.bcolind[jj];
			const index loccol = jj - mat.browptr[irow];
			const index saicolpos = tsp.bcolptr[startcol+loccol];

			int numentries = 0;

			for(index kk = mat.browptr[colind]; kk < mat.browendptr[colind]; kk++)
			{
				for(index ll = mat.browptr[irow]; ll < mat.browendptr[irow]; ll++) {
					if(mat.bcolind[ll] == mat.bcolind[kk])
					{
						tsp.bpos[saicolpos+numentries] = kk;
						tsp.browind[saicolpos+numentries] = ll-mat.browptr[irow];
						numentries++;
					}
				}
			}
		}
	}

	return tsp;
}

template LeftSAIPattern<int>
left_incomplete_SAI_pattern(const SRMatrixStorage<const double,const int>&& mat);

/// Storage type for the left-hand side matrix for computing the left SAI/ISAI corresponding to one row
template <typename scalar>
using LMatrix = Matrix<scalar,Dynamic,Dynamic,ColMajor>;

/// Gather the SAI/ISAI LHS operator for one (block-)row of the matrix, given the SAI/ISAI pattern
template <typename scalar, typename index, int bs, StorageOptions stor>
LMatrix<scalar> gather_lhs_matrix(const SRMatrixStorage<const scalar,const index>& mat,
                                  const LeftSAIPattern<index>& sp,
                                  const index row);

/// Scatter the SAI/ISAI solution for one (block-)row of the approx inverse, given the pattern
template <typename scalar, typename index, int bs, StorageOptions stor>
void scatter_solution(const LMatrix<scalar>& sol, const LeftSAIPattern<index>& sp,
                      const index row, SRMatrixStorage<scalar,index>& mat);

template <typename scalar, typename index, int bs, StorageOptions stor>
void compute_SAI(const SRMatrixStorage<const scalar,const index>& mat, const LeftSAIPattern<index>& sp,
                 const int thread_chunk_size, const bool fullsai,
                 SRMatrixStorage<scalar,index>& sai)
{
	assert(mat.nbrows == sai.nbrows);
	assert(mat.nnzb == sai.nnzb);

	// Solve least-squares problem for each row
#pragma omp parallel for default(shared) schedule(dynamic,thread_chunk_size)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		const LMatrix<scalar> lhs = gather_lhs_matrix<scalar,index,bs,stor>(mat, sp, irow);
		LMatrix<scalar> rhs = LMatrix<scalar>::Zero(lhs.rows(), bs);
		for(int i = 0; i < bs; i++)
			rhs(sp.localCentralRow[irow]*bs + i, i) = 1.0;

		LMatrix<scalar> ans;
		if (fullsai)
			ans = lhs.colPivHouseholderQr().solve(rhs);
		else
			ans = lhs.partialPivLu().solve(rhs);

		scatter_solution<scalar,index,bs,stor>(ans, sp, irow, sai);
	}
}

template void compute_SAI<double,int,1,ColMajor>(const SRMatrixStorage<const double,const int>& mat,
                                                 const LeftSAIPattern<int>& sp,
                                                 const int thread_chunk_size, const bool fullsai,
                                                 SRMatrixStorage<double,int>& sai);
template void compute_SAI<double,int,4,ColMajor>(const SRMatrixStorage<const double,const int>& mat,
                                                 const LeftSAIPattern<int>& sp,
                                                 const int thread_chunk_size, const bool fullsai,
                                                 SRMatrixStorage<double,int>& sai);

template <typename scalar, typename index, int bs, StorageOptions stor>
LMatrix<scalar> gather_lhs_matrix(const SRMatrixStorage<const scalar,const index>& mat,
                                  const LeftSAIPattern<index>& sp,
                                  const index row)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *const valb = reinterpret_cast<const Blk*>(&mat.vals[0]);

	LMatrix<scalar> lhs = LMatrix<scalar>::Zero(sp.nEqns[row]*bs, sp.nVars[row]*bs);

	for(int jcol = 0; jcol < sp.nVars[row]; jcol++)
	{
		const index start = sp.bcolptr[sp.sairowptr[row]+jcol],
			end = sp.bcolptr[sp.sairowptr[row]+jcol+1];

		assert(end-start == sp.nEqns[row]);

#pragma omp simd
		for(index ii = start; ii < end; ii++)
		{
			const int lhsrow = sp.browind[ii];
			const index lhspos = sp.bpos[ii];

			// Note the transpose
			for(int ib = 0; ib < bs; ib++)
				for(int jb = 0; jb < bs; jb++)
					lhs(lhsrow*bs+ib, jcol*bs+jb) = valb[lhspos](jb,ib);
		}
	}

	return lhs;
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void scatter_solution(const LMatrix<scalar>& sol, const LeftSAIPattern<index>& sp,
                      const index row, SRMatrixStorage<scalar,index>& saimat)
{
	using Blk = Block_t<scalar,bs,stor>;
	Blk *const saivalb = reinterpret_cast<Blk*>(&saimat.vals[0]);

	assert(sp.nVars[row] == saimat.browendptr[row] - saimat.browptr[row]);

	for(index jj = saimat.browptr[row]; jj < saimat.browendptr[row]; jj++)
	{
		const int locloc = jj - saimat.browptr[row];

		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
				saivalb[jj](i,j) = sol(locloc*bs+j, i);
	}
}

}
