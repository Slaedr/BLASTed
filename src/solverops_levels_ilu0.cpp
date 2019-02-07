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
#include "solverops_ilu0.hpp"
#include "async_ilu_factor.hpp"
#include "async_blockilu_factor.hpp"

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

