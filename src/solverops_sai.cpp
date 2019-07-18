/** \file
 * \brief Sparse approximate preconditioner implementation
 */

#include "sai.hpp"
#include "solverops_sai.hpp"

namespace blasted {

template <typename scalar, typename index>
struct LeftSAIImpl
{
	LeftSAIPattern<index> sp;             ///< Pattern for gathering the row-wise SAI matrices
	RawBSRMatrix<scalar,index> saimat;    ///< The sparse approximate inverse
};

template <typename scalar, typename index, int bs, StorageOptions stor>
LeftSAIPreconditioner<scalar,index,bs,stor>::LeftSAIPreconditioner()
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
void LeftSAIPreconditioner<scalar,index,bs,stor>::compute()
{
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void LeftSAIPreconditioner<scalar,index,bs,stor>::apply(const scalar *const x,
                                                        scalar *const __restrict y) const
{
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void LeftSAIPreconditioner<scalar,index,bs,stor>::apply_relax(const scalar *const x,
                                                              scalar *const __restrict y) const
{
	throw std::runtime_error("SAI does not provide relaxation!");
}

template class LeftSAIPreconditioner<double,int,1,ColMajor>;
template class LeftSAIPreconditioner<double,int,4,ColMajor>;

}
