/** \file
 * \brief Implementation of some functionality useful for building SAI-type preconditioners
 */

#include <set>
#include "sai.hpp"
#include "helper_algorithms.hpp"

namespace blasted {

template <typename scalar, typename index>
LeftSAIPattern<index> left_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat)
{
	LeftSAIPattern<index> tsp;

	/* Note that every block-row of A corresponds to a block-row of the approx inverse M and to a
	 * least-squares problem.
	 */

	tsp.sairowptr.assign(mat.nbrows+1,0);
	tsp.nVars.assign(mat.nbrows,0);
	tsp.nEqns.assign(mat.nbrows,0);

	// Step 1: Compute number of variables and constraints for each least-squares problem

	index totalcoeffs = 0;     // number of least-square LHS coeffs over all block-rows
	//#pragma NOT PARALLEL omp parallel for default(shared) reduction(+:totalcoeffs)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		tsp.nVars[irow] = mat.browendptr[irow] - mat.browptr[irow];

		std::set<index> constraints;
		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A

			for(index kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
				constraints.insert(mat.bcolind[kk]);
		}

		tsp.nEqns[irow] = static_cast<int>(constraints.size());
		totalcoeffs += tsp.nVars[irow]*tsp.nEqns[irow];
		tsp.sairowptr[irow+1] = tsp.nVars[irow];
	}

	// get the starting of each least-squares problem in bcolptr
	internal::inclusive_scan(tsp.sairowptr);

	const index totalvars = tsp.sairowptr[mat.nbrows];
	tsp.bcolptr.assign(totalvars,0);
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

		std::vector<std::vector<int>> localrowinds(tsp.nVars[irow]);

		for(index jj = mat.browptr[irow]; jj < mat.browendptr[irow]; jj++)
		{
			const index col = mat.bcolind[jj];
			const int localcolidx = jj - mat.browptr[irow];
			localrowinds[localcolidx].assign(mat.browendptr[col]-mat.browptr[col], -1);

			for(size_t i = 0; i < constrvec.size(); i++)
			{
				bool found = false;

				for(int kk = mat.browptr[col]; kk < mat.browendptr[col]; kk++)
					if(constrvec[i] == mat.bcolind[kk])
					{
						localrowinds[localcolidx][kk-mat.browptr[col]] = i;
						found = true;
					}

				if(!found)
					throw std::runtime_error("SAI pattern: Not found local row index!");
			}
		}

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
				const int locpos = mat.browptr[col]-kk;
				tsp.bpos[tsp.bcolptr[tsp.sairowptr[irow]+localcolidx] + locpos] = kk;
				tsp.browind[tsp.bcolptr[tsp.sairowptr[irow]+localcolidx] + locpos]
					= localrowinds[localcolidx][kk];
			}
		}
	}

	return tsp;
}

template LeftSAIPattern<int> left_SAI_pattern(const CRawBSRMatrix<double,int>& mat);

// ******************** OLD STUFF *************************

#if 0
template <typename scalar, typename index>
TriangularLeftSAIPattern<index> triangular_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat)
{
	TriangularLeftSAIPattern<index> tsp;

	// compute number of columns of A that are needed for every row of M

	tsp.ptrlower.resize(mat.nbrows+1);
	tsp.ptrupper.resize(mat.nbrows+1);
	tsp.ptrlower[0] = 0;
	tsp.ptrupper[0] = 0;
	tsp.lowerMConstraints(mat.nbrows);
	tsp.upperMConstraints(mat.nbrows);

	//#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		const index lowerNVars = mat.diagind[irow]+1 - mat.browptr[irow];

		std::set<index> lcols;
		for(index jj = mat.browptr[irow]; jj <= mat.diagind[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A

			for(index kk = mat.browptr[col]; kk <= mat.diagind[col]; kk++)
				lcols.insert(mat.bcolind[kk]);
		}

		tsp.lowerMConstraints[irow] = static_cast<int>(lcols.size());
		tsp.ptrlower[irow+1] = tsp.lowerMConstraints[irow]*lowerNVars;

		const index upperNVars = mat.browptr[irow+1] - mat.diagind[irow];

		std::set<index> ucols;
		for(index jj = mat.diagind[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A

			for(index kk = mat.diagind[col]; kk < mat.browptr[col+1]; kk++)
				ucols.insert(mat.bcolind[kk]);
		}

		tsp.upperMConstraints[irow] = static_cast<int>(ucols.size());
		tsp.ptrupper[irow+1] = tsp.upperMConstraints[irow]*upperNVars;
	}

	internal::inclusive_scan(tsp.ptrlower);
	internal::inclusive_scan(tsp.ptrupper);

	lowernz.resize(tsp.ptrlower[mat.nbrows]);
	uppernz.resize(tsp.ptrupper[mat.nbrows]);

	// compute locations of L that are needed for every row of the approx inverse M

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		std::set<index> lcols;
		std::vector<index> lrows;
		for(index jj = mat.browptr[irow]; jj <= mat.diagind[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			lrows.push_back(col);

			for(index kk = mat.browptr[col]; kk <= mat.diagind[col]; kk++)
				lcols.insert(mat.bcolind[kk]);
		}

		const int ml = tsp.lowerNConstraints[irow];
		const int nl = mat.diagind[irow]+1 - mat.browptr[irow];

		for(int i = 0; i < ml*nl; i++)
			tsp.lowernz[tsp.ptrlower[irow]+i] = -1;

		for(int j = 0; j < nl; j++)
		{
			const int lloccol = j;
			const index rowL = lrows[j];

			int llocrow = 0;
			for(auto icol = lcols.begin(); icol != lcols.end(); icol++) {
				for(index kk = mat.browptr[rowL]; kk <= mat.diagind[rowL]; kk++) {
					if(*icol == mat.bcolind[kk])
					{
						tsp.lowernz[tsp.ptrlower[irow] + llocrow + lloccol*ml] = kk;
					}
				}
				llocrow++;
			}
			assert(llocrow == ml);
		}
	}

	// compute locations of U that are needed for every row of the approx inverse M

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		std::set<index> ucols;
		std::vector<index> urows;
		for(index jj = mat.diagind[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			urows.push_back(col);

			for(index kk = mat.diagind[col]; kk < mat.browptr[col+1]; kk++)
				ucols.insert(mat.bcolind[kk]);
		}

		const int ml = tsp.upperNConstraints[irow];
		const int nl = mat.browptr[irow+1] - mat.diagind[irow];

		for(int i = 0; i < ml*nl; i++)
			tsp.uppernz[tsp.ptrupper[irow]+i] = -1;

		for(int j = 0; j < nl; j++)
		{
			const int uloccol = j;
			const index rowU = urows[j];

			int ulocrow = 0;
			for(auto icol = ucols.begin(); icol != ucols.end(); icol++) {
				for(index kk = mat.diagind[rowU]; kk < mat.browptr[rowU+1]; kk++) {
					if(*icol == mat.bcolind[kk])
					{
						tsp.uppernz[tsp.ptrupper[irow] + ulocrow + uloccol*ml] = kk;
					}
				}
				ulocrow++;
			}
			assert(ulocrow == ml);
		}
	}

	return tsp;
}

template TriangularLeftSAIPattern<int> triangular_SAI_pattern(const CRawBSRMatrix<double,int>& mat);

template <typename scalar, typename index>
TriangularLeftSAIPattern<index> triangular_incomp_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat)
{
	TriangularLeftSAIPattern<index> tsp;

	// compute number of columns of A that are needed for every row of M

	tsp.ptrlower.resize(mat.nbrows+1);
	tsp.ptrupper.resize(mat.nbrows+1);
	tsp.ptrlower[0] = 0;
	tsp.ptrupper[0] = 0;
	tsp.lowerMConstraints(mat.nbrows);
	tsp.upperMConstraints(mat.nbrows);

	//#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		const index lowerNVars = mat.diagind[irow]+1 - mat.browptr[irow];
		tsp.ptrlower[irow+1] = lowerNVars*lowerNVars;

		const index upperNVars = mat.browptr[irow+1] - mat.diagind[irow];
		tsp.ptrupper[irow+1] = tsp.upperMConstraints[irow]*upperNVars;
	}

	internal::inclusive_scan(tsp.ptrlower);
	internal::inclusive_scan(tsp.ptrupper);

	lowernz.resize(tsp.ptrlower[mat.nbrows]);
	uppernz.resize(tsp.ptrupper[mat.nbrows]);

	// compute locations of L that are needed for every row of the approx inverse M

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		std::vector<index> lrows;
		for(index jj = mat.browptr[irow]; jj <= mat.diagind[irow]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			lrows.push_back(col);
		}

		const int nl = mat.diagind[irow]+1 - mat.browptr[irow];

		for(int i = 0; i < nl*nl; i++)
			tsp.lowernz[tsp.ptrlower[irow]+i] = -1;

		for(int j = 0; j < nl; j++)
		{
			const index rowL = lrows[j];

			for(int i = 0; i < nl; i++)
			{
				const index colL = lrows[i];
				for(index kk = mat.browptr[rowL]; kk <= mat.diagind[rowL]; kk++) {
					if(mat.bcolind[kk] == colL)
						tsp.lowernz[tsp.ptrlower[irow] + i + j*nl] = kk;
				}
			}
		}
	}

	// compute locations of U that are needed for every row of the approx inverse M

	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		std::vector<index> urows;
		for(index jj = mat.diagind[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			const index col = mat.bcolind[jj];      // column of A^T, row of A
			urows.push_back(col);
		}

		const int nl = mat.browptr[irow+1] - mat.diagind[irow];

		for(int i = 0; i < nl*nl; i++)
			tsp.uppernz[tsp.ptrupper[irow]+i] = -1;

		for(int j = 0; j < nl; j++)
		{
			const index rowU = urows[j];

			for(int i = 0; i < nl; i++)
			{
				const index colU = urows[i];
				for(index kk = mat.diagind[rowU]; kk < mat.browptr[rowU+1]; kk++) {
					if(colU == mat.bcolind[kk])
						tsp.uppernz[tsp.ptrupper[irow] + i + j*nl] = kk;
				}
			}
		}
	}

	return tsp;
}

template TriangularLeftSAIPattern<int>
triangular_incomp_SAI_pattern<double,int>(const CRawBSRMatrix<double,int>& mat);
#endif

}
