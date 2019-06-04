/** \file
 * \brief Implementation of some functionality useful for building SAI-type preconditioners
 */

#include <set>
#include "sai.hpp"
#include "helper_algorithms.hpp"

namespace blasted {

template <typename scalar, typename index>
TriangularLeftSAIPattern<index> compute_triangular_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat,
                                                               const bool fullsai)
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
						tsp.lowernz[tsp.ptrlower[irow] + llocrow + lloccol*nl] = kk;
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
						tsp.uppernz[tsp.ptrupper[irow] + ulocrow + uloccol*nl] = kk;
					}
				}
				ulocrow++;
			}
			assert(ulocrow == ml);
		}
	}

	return tsp;
}

}
