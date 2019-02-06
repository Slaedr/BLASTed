/** \file
 * \brief Level-scheduling algorithms
 * \author Aditya Kashi
 */

#ifndef BLASTED_LEVEL_SCHEDULE_H
#define BLASTED_LEVEL_SCHEDULE_H

#include <vector>
#include "srmatrixdefs.hpp"

namespace blasted {

/// Divides the unknowns in a system into contiguous and ordered levels
/** \param mat The original matrix. It is assumed that column indices in each row are ordered.
 * \return A list containing, for each level, the starting (block)row index for that level. There is one
 * extra entry in the end for convenience - this is always one past the total number of (block-) rows.
 */
template <typename scalar, typename index>
std::vector<index> computeLevels(const CRawBSRMatrix<scalar,index>& mat);

}

#endif
