/** \file kernels_relaxation.hpp
 * \brief Kernels useful for linear relaxation solvers
 * \author Aditya Kashi
 */

#ifndef BLASTED_KERNELS_RELAXATION_H
#define BLASTED_KERNELS_RELAXATION_H

#include "kernels_base.hpp"
#include "../impldefs.hpp"

namespace blasted {

/// Relax one block-row
/** Note that there is no aliasing issue between xL,xU and y.
 * BUT there IS an aliasing issue between rhs and y.
 */
template <typename scalar, typename index, int bs, class Mattype, class Vectype> inline
void block_relax
	(Map<const Mattype>& vals, const index *const __restrict bcolind,
	 const index irow, const index browstart, const index bdiagind, const index nextbrowstart,
	 Map<const Mattype>& diaginv, Map<const Vector<scalar>>& rhs, 
	 Map<const Vector<scalar>>& xL, Map<const Vector<scalar>>& xU,
	 Map<Vector<scalar>>& y
	 )
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0)*xL.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs)*xL.SEG<bs>(bcolind[jj]*bs);
	
	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
		if(Mattype::IsRowMajor)
			inter += vals.BLK<bs,bs>(jj*bs,0)*xU.SEG<bs>(bcolind[jj]*bs);
		else
			inter += vals.BLK<bs,bs>(0,jj*bs)*xU.SEG<bs>(bcolind[jj]*bs);

	if(Mattype::IsRowMajor)
		y.SEG<bs>(irow*bs).noalias() = diaginv.BLK<bs,bs>(irow*bs,0) 
											  * (rhs.SEG<bs>(irow*bs) - inter);
	else
		y.SEG<bs>(irow*bs).noalias() = diaginv.BLK<bs,bs>(0,irow*bs) 
											  * (rhs.SEG<bs>(irow*bs) - inter);
}

/// Relax one block-row using the new kernel
/** Note that there is no aliasing issue between xL,xU and y.
 * BUT there IS an aliasing issue between rhs and y.
 */
template <typename scalar, typename index, int bs, StorageOptions stor> inline
void block_relax_kernel
(const Block_t<scalar,bs,stor> *const vals,
 const index *const __restrict bcolind,
 const index irow, const index browstart, const index bdiagind,
 const index nextbrowstart, const Block_t<scalar,bs,stor>& diaginv,
 const Segment_t<scalar,bs>& rhs,
 const Segment_t<scalar,bs> *const xL, const Segment_t<scalar,bs> *const xU,
 Segment_t<scalar,bs>& y
 )
{
	Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

	for(index jj = browstart; jj < bdiagind; jj++)
	  inter += vals[jj]*xL[bcolind[jj]];

	for(index jj = bdiagind+1; jj < nextbrowstart; jj++)
	  inter += vals[jj]*xU[bcolind[jj]];

	y.noalias() = diaginv * (rhs - inter);
}

/// Relax one row
template <typename scalar, typename index> inline 
scalar scalar_relax(const scalar *const vals, const index *const colind, 
                    const index rowstart, const index diagind, const index nextrowstart,
                    const scalar diag_entry_inv, const scalar rhs,
                    const scalar *const xL, const scalar *const xU)
{
	scalar inter = 0;
	for(index jj = rowstart; jj < diagind; jj++)
		inter += vals[jj]*xL[colind[jj]];

	for(index jj = diagind+1; jj < nextrowstart; jj++)
		inter += vals[jj]*xU[colind[jj]];

	return diag_entry_inv * (rhs - inter);
}

}

#endif
