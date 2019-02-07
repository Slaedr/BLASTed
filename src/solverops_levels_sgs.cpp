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
#include "levelschedule.hpp"
#include "solverops_levels_sgs.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stor>
Level_BSGS<scalar,index,bs,stor>::Level_BSGS() : BJacobiSRPreconditioner<scalar,index,bs,stor>()
{ }

}
