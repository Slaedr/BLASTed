/** \file
 * \brief Implementation of matrix-matrix product for sparse-row matrices
 */

#include "matmat.hpp"

namespace blasted {

template <typename scalar, typename index, int bs, StorageOptions stor>
SRMatrixStorage<scalar,index>
computeMatMatLUSparsityPattern(const SRMatrixStorage<const scalar,const index>&& ilu)
{
	ArrayView<index> browptr(ilu.nbrows);
}

}
