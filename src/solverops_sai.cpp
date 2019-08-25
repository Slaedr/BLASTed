/** \file
 * \brief Sparse approximate preconditioner implementation
 */

#include "blas/matvecs.hpp"
#include "sai.hpp"
#include "solverops_sai.hpp"

namespace blasted {

template <typename scalar, typename index>
struct LeftSAIImpl
{
	LeftSAIPattern<index> sp;             ///< Pattern for gathering the row-wise SAI matrices
	SRMatrixStorage<scalar,index> saimat; ///< The sparse approximate inverse
};

template <typename scalar, typename index, int bs, StorageOptions stor>
LeftSAIPreconditioner<scalar,index,bs,stor>
::LeftSAIPreconditioner(SRMatrixStorage<const scalar,const index>&& matrix, const int tcs)
	: SRPreconditioner<scalar,index>(std::move(matrix)), thread_chunk_size{tcs}
{
	impl.sp = left_SAI_pattern(static_cast<const SRMatrixStorage<const scalar,const index>&&>(pmat));
}

template <typename scalar, typename index, int bs, StorageOptions stor>
PrecInfo LeftSAIPreconditioner<scalar,index,bs,stor>::compute()
{
	compute_SAI<scalar,index,bs,stor>(pmat, impl.sp, thread_chunk_size, true, impl.saimat);
	return PrecInfo();
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void LeftSAIPreconditioner<scalar,index,bs,stor>::apply(const scalar *const x,
                                                        scalar *const __restrict y) const
{
	BLAS_BSR<scalar,index,bs,stor>
		::matrix_apply(static_cast<const SRMatrixStorage<scalar,index>&&>(impl.saimat),
		               x, y);
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
