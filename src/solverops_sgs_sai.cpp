/** \file
 * \brief Implementation of SGS (I)SAI
 */

#include <boost/align/aligned_alloc.hpp>
#include <Eigen/Dense>
#include "solverops_sgs_sai.hpp"

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

template <typename scalar, typename index, int bs, StorageOptions stopt>
BSGS_SAI<scalar,index,bs,stopt>::BSGS_SAI(const bool full_spai)
	: fullsai{full_spai}, sai.nbrows{0}, sai.browptr{nullptr}, sai.bcolind{nullptr},
	sai.diagind{nullptr}, sai.vals{nullptr}
{ }

template <typename scalar, typename index, int bs, StorageOptions stopt>
BSGS_SAI<scalar,index,bs,stopt>::~BSGS_SAI()
{
	aligned_free(sai.browptr);
	aligned_free(sai.bcolind);
	aligned_free(sai.vals);
	aligned_free(sai.diagind);
}

template <typename scalar, typename index, int bs, StorageOptions stopt>
void BSGS_SAI<scalar,index,bs,stopt>::compute()
{
	if(mat.nbrows == 0 || mat.browptr == nullptr)
		throw std::runtime_error("BSGS_SAI: Matrix not set!");

	/// Assume sparsity pattern does not change once set
	if(!sai.browptr) {
		sai.nbrows = mat.nbrows;
		sai.browptr = (index*)aligned_alloc(CACHE_LINE_LEN, (mat.nbrows+1)*sizeof(index));
		sai.bcolind = (index*)aligned_alloc(CACHE_LINE_LEN, mat.browptr[mat.nbrows]*sizeof(index));
		sai.vals = (scalar*)aligned_alloc(CACHE_LINE_LEN, mat.browptr[mat.nbrows]*bs*bs*sizeof(scalar));
		sai.diagind = (index*)aligned_alloc(CACHE_LINE_LEN, mat.nbrows*sizeof(index));

		for(index i = 0; i < mat.nbrows; i++) {
			sai.browptr[i] = mat.browptr[i];
			sai.diagind[i] = mat.diagind[i];
		}
		sai.browptr[mat.nbrows] = mat.browptr[mat.nbrows];

		for(index jj = 0; jj < mat.browptr[mat.nbrows]; jj++) {
			sai.bcolind[jj] = mat.bcolind[jj];
		}
	}

	// TODO: Compute SAI or ISAI
}

template <typename scalar, typename index, int bs, StorageOptions stopt>
void apply(const scalar *const x, scalar *const __restrict y) const
{
	bsr_matrix_apply<scalar,index,bs,stopt>(sai, x, y);
}

template <typename scalar, typename index, int bs, StorageOptions stopt>
void apply_relax(const scalar *const x, scalar *const __restrict y) const
{
	throw std::runtime_error("BSGS_SAI does not have relaxation!");
}

}
