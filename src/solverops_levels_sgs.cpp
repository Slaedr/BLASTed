/** \file
 * \brief Implementation of level-scheduled Gauss-Seidel
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

#include <boost/align/aligned_alloc.hpp>
#include "kernels/kernels_sgs.hpp"
#include "kernels/kernels_relaxation.hpp"
#include "levelschedule.hpp"
#include "solverops_levels_sgs.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
Level_BSGS<scalar,index,bs,stor>::Level_BSGS()
	: BJacobiSRPreconditioner<scalar,index,bs,stor>(), ytemp{nullptr}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
Level_BSGS<scalar,index,bs,stor>::~Level_BSGS()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Level_BSGS<scalar,index,bs,stor>::compute()
{
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*bs*sizeof(scalar));
		levels = computeLevels(&mat);
	}

	BJacobiSRPreconditioner<scalar,index,bs,stor>::compute();
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Level_BSGS<scalar,index,bs,stor>::apply(const scalar *const rr,
                                             scalar *const __restrict zz) const
{
	const Block_t<scalar,bs,stor> *const mvals
		= reinterpret_cast<const Block_t<scalar,bs,stor>*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(ytemp);

	const index nlevels = static_cast<index>(levels.size())-1;

	// forward solve
	for(index ilvl = 0; ilvl < nlevels; ilvl++)
	{
#pragma omp parallel for default(shared)
		for(index irow = levels[ilvl]; irow < levels[ilvl+1]; irow++)
		{
			kernels::block_fgs<scalar,index,bs,stor>(mvals, mat.bcolind, irow, mat.browptr[irow], 
			                                         mat.diagind[irow], dblks[irow], r[irow], y);
		}
	}

	// backward solve
	for(index ilvl = nlevels; ilvl >= 1; ilvl--)
	{
#pragma omp parallel for default(shared)
		for(index irow = levels[ilvl]-1; irow >= levels[ilvl-1]; irow--)
		{
			kernels::block_bgs<scalar,index,bs,stor>(mvals, mat.bcolind, irow, mat.diagind[irow],
			                                         mat.browptr[irow+1], dblks[irow], y[irow], z);
		}
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Level_BSGS<scalar,index,bs,stor>::apply_relax(const scalar *const bb,
                                                   scalar *const __restrict xx) const
{
	const Blk *mvals = reinterpret_cast<const Blk*>(mat.vals);
	const Blk *dblks = reinterpret_cast<const Blk*>(dblocks);
	const Seg *b = reinterpret_cast<const Seg*>(bb);
	// the solution vector is wrapped in both a pointer to const segment and one to mutable segment
	const Seg *x = reinterpret_cast<const Seg*>(xx);
	Seg *xmut = reinterpret_cast<Seg*>(xx);

	const index nlevels = static_cast<index>(levels.size())-1;

	for(int step = 0; step < solveparams.maxits; step++)
	{
		for(index ilvl = 0; ilvl < nlevels; ilvl++) {
#pragma omp parallel for default(shared)
			for(index irow = levels[ilvl]; irow < levels[ilvl+1]; irow++)
			{
				block_relax_kernel<scalar,index,bs,stor>
					(mvals, mat.bcolind, irow,
					 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
					 dblks[irow], b[irow], x, x, xmut[irow]);
			}
		}

		for(index ilvl = nlevels; ilvl >= 1; ilvl--) {
#pragma omp parallel for default(shared)
			for(index irow = levels[ilvl]-1; irow >= levels[ilvl-1]; irow--)
			{
				block_relax_kernel<scalar,index,bs,stor>
					(mvals, mat.bcolind, irow,
					 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
					 dblks[irow], b[irow], x, x, xmut[irow]);
			}
		}
	}
}

template class Level_BSGS<double,int,4,ColMajor>;
template class Level_BSGS<double,int,5,ColMajor>;
template class Level_BSGS<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class Level_BSGS<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class Level_BSGS<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

template <typename scalar, typename index>
Level_SGS<scalar,index>::Level_SGS() : JacobiSRPreconditioner<scalar,index>(), ytemp{nullptr}
{ }

template <typename scalar, typename index>
Level_SGS<scalar,index>::~Level_SGS()
{
	aligned_free(ytemp);
}

template <typename scalar, typename index>
void Level_SGS<scalar,index>::compute()
{
	if(!ytemp) {
		ytemp = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(scalar));
		levels = computeLevels(&mat);
	}

	JacobiSRPreconditioner<scalar,index>::compute();
}

template <typename scalar, typename index>
void Level_SGS<scalar,index>::apply(const scalar *const rr,
                                    scalar *const __restrict zz) const
{
	const index nlevels = static_cast<index>(levels.size())-1;

	// forward solve
	for(index ilvl = 0; ilvl < nlevels; ilvl++)
	{
#pragma omp parallel for default(shared)
		for(index irow = levels[ilvl]; irow < levels[ilvl+1]; irow++)
		{
			ytemp[irow] = kernels::scalar_fgs(mat.vals, mat.bcolind,
			                                  mat.browptr[irow], mat.diagind[irow],
			                                  dblocks[irow], rr[irow], ytemp);
		}
	}

	// backward solve
	for(index ilvl = nlevels; ilvl >= 1; ilvl--)
	{
#pragma omp parallel for default(shared)
		for(index irow = levels[ilvl]-1; irow >= levels[ilvl-1]; irow--)
		{
			zz[irow] = kernels::scalar_bgs(mat.vals, mat.bcolind,
			                               mat.diagind[irow], mat.browptr[irow+1],
			                               mat.vals[mat.diagind[irow]], dblocks[irow],
			                               ytemp[irow], zz);
		}
	}
}

template <typename scalar, typename index>
void Level_SGS<scalar,index>::apply_relax(const scalar *const b,
                                          scalar *const __restrict x) const
{
	const index nlevels = static_cast<index>(levels.size())-1;

	for(int step = 0; step < solveparams.maxits; step++)
	{
		for(index ilvl = 0; ilvl < nlevels; ilvl++) {
#pragma omp parallel for default(shared)
			for(index irow = levels[ilvl]; irow < levels[ilvl+1]; irow++)
			{
				x[irow] = scalar_relax<scalar,index>
					(mat.vals, mat.bcolind,
					 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
					 dblocks[irow], b[irow], x, x);
			}
		}

		for(index ilvl = nlevels; ilvl >= 1; ilvl--) {
#pragma omp parallel for default(shared)
			for(index irow = levels[ilvl]-1; irow >= levels[ilvl-1]; irow--)
			{
				x[irow] = scalar_relax<scalar,index>
					(mat.vals, mat.bcolind, 
					 mat.browptr[irow], mat.diagind[irow], mat.browptr[irow+1],
					 dblocks[irow], b[irow], x, x);
			}
		}
	}
}

template class Level_SGS<double,int>;

}
