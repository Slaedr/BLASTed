/** \file
 * \brief Implementation of ILU-type iterations using level-scheduled factorization and/or application
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

#include <iostream>
#include <boost/align/aligned_alloc.hpp>
#include "kernels/kernels_ilu_apply.hpp"
#include "levelschedule.hpp"
#include "solverops_levels_ilu0.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
Async_Level_BlockILU0<scalar,index,bs,stor>
::Async_Level_BlockILU0(const int nbuildsweeps, const int thread_chunk_size,
                        const FactInit fact_inittype, const bool threadedfactor,
                        const bool compute_remainder)
	: AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>(nbuildsweeps,1,thread_chunk_size,
	                                                        fact_inittype, INIT_A_NONE, threadedfactor,
	                                                        true, compute_remainder)
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
Async_Level_BlockILU0<scalar,index,bs,stor>::~Async_Level_BlockILU0()
{
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Async_Level_BlockILU0<scalar,index,bs,stor>::compute()
{
	if(!iluvals)
		levels = computeLevels(mat);

	AsyncBlockILU0_SRPreconditioner<scalar,index,bs,stor>::compute();
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Async_Level_BlockILU0<scalar,index,bs,stor>::apply(const scalar *const rr, 
                                                        scalar *const __restrict zz) const
{
	const Blk *ilu = reinterpret_cast<const Blk*>(iluvals);
	const Seg *r = reinterpret_cast<const Seg*>(rr);
	Seg *z = reinterpret_cast<Seg*>(zz);
	Seg *y = reinterpret_cast<Seg*>(ytemp);

	const index nlevels = static_cast<index>(levels.size())-1;

	for(int ilvl = 0; ilvl < nlevels; ilvl++)
	{
#pragma omp parallel for default(shared)
		for(index i = levels[ilvl]; i < levels[ilvl+1]; i++)
		{
			block_unit_lower_triangular<scalar,index,bs,stor>
				(ilu, mat.bcolind, mat.browptr[i], mat.diagind[i], r[i], i, y);
		}
	}

	for(int ilvl = nlevels; ilvl > 0; ilvl--)
	{
#pragma omp parallel for default(shared)
		for(index i = levels[ilvl]-1; i >= levels[ilvl-1]; i--)
		{
			block_upper_triangular<scalar,index,bs,stor>
				(ilu, mat.bcolind, mat.diagind[i], mat.browptr[i+1], y[i], i, z);
		}
	}
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void Async_Level_BlockILU0<scalar,index,bs,stor>::apply_relax(const scalar *const r, 
                                                              scalar *const __restrict z) const
{
	throw std::runtime_error("ILU relaxation not implemented!");
}

template class Async_Level_BlockILU0<double,int,4,ColMajor>;
template class Async_Level_BlockILU0<double,int,5,ColMajor>;
template class Async_Level_BlockILU0<double,int,4,RowMajor>;

#ifdef BUILD_BLOCK_SIZE
template class Async_Level_BlockILU0<double,int,BUILD_BLOCK_SIZE,ColMajor>;
template class Async_Level_BlockILU0<double,int,BUILD_BLOCK_SIZE,RowMajor>;
#endif

}
