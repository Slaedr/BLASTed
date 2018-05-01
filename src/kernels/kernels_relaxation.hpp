/** \file kernels_relaxation.hpp
 * \brief Kernels useful for linear relaxation solvers
 * \author Aditya Kashi
 */

#ifndef BLASTED_KERNELS_RELAXATION_H
#define BLASTED_KERNELS_RELAXATION_H

#include "kernels_base.hpp"

namespace blasted {

/// Relax one block-row
template <typename scalar, typename index, int bs, class Mattype, class Vectype> inline
void block_relax
	(Map<const Mattype>& vals, const index *const __restrict bcolind,
	 const index irow, const index browstart, const index bdiagind, const index nextbrowstart,
	 Map<const Mattype>& diaginv, Map<const Vector<scalar>>& rhs, 
	 Map<const Vector<scalar>>& xL, Map<const Vector<scalar>>& xU,
	 Map<Vector<scalar>>& y
	 /*const MatrixBase<Vectype>& xL, const MatrixBase<Vectype>& xU,
	   MatrixBase<Vectype>& y*/
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

}

#endif
