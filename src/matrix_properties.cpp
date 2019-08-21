/** \file
 * \brief Implementation of computation of some properties of local matrices
 */

#include "matrix_properties.hpp"

namespace blasted {

/// Uses one kernel to compute all 4 reductions - L_avg, L_min, U_avg, U_min diagonal dominance
template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,4> diagonal_dominance(const SRMatrixStorage<const scalar,const index>&& mat)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);

	scalar uddavg = 0, uddmin = 1e30, lddavg = 0, lddmin = 1e30;

#pragma omp parallel for default(shared) reduction(+:uddavg,lddavg) reduction(min:uddmin,lddmin)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		scalar rowddu[bs], rowddl[bs];
		for(int i = 0; i < bs; i++) {
			rowddl[i] = 0;
			rowddu[i] = 0;
		}

		const index diagp = mat.diagind[irow];

		// add the off-diagonal entries of the diagonal block
		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
				if(i != j)
					rowddu[i] += std::abs(data[diagp](i,j));

		// other off-diagonal entries for upper
		for(index jj = diagp+1; jj < mat.browendptr[irow]; jj++)
		{
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					rowddu[i] += std::abs(data[jj](i,j));
		}

		// off-diagonal entries for lower
		for(index jj = mat.browptr[irow]; jj < diagp; jj++)
		{
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					rowddl[i] += std::abs(data[jj](i,j));
		}

		for(int i = 0; i < bs; i++) {
			rowddl[i] = 1.0 - rowddl[i];                             //< lower
			rowddu[i] = 1.0 - rowddu[i]/std::abs(data[diagp](i,i));  //< upper
		}

		scalar uavg_blk = 0, lavg_blk = 0, umin_blk = 1e30, lmin_blk = 1e30;
		for(int i = 0; i < bs; i++)
		{
			if(umin_blk > rowddu[i])
				umin_blk = rowddu[i];
			if(lmin_blk > rowddl[i])
				lmin_blk = rowddl[i];

			lavg_blk += rowddl[i];
			uavg_blk += rowddu[i];
		}

		if(uddmin > umin_blk)
			uddmin = umin_blk;
		if(lddmin > lmin_blk)
			lddmin = lmin_blk;
		uddavg += uavg_blk;
		lddavg += lavg_blk;
	}

	return {lddavg/(mat.nbrows*bs), lddmin, uddavg/(mat.nbrows*bs), uddmin};
}

template std::array<double,4>
diagonal_dominance<double,int,1,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,4>
diagonal_dominance<double,int,4,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,4>
diagonal_dominance<double,int,5,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,4>
diagonal_dominance<double,int,4,RowMajor>(const SRMatrixStorage<const double,const int>&& mat);

template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,2> diagonal_dominance_upper(const SRMatrixStorage<const scalar,const index>&& mat)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);

	scalar ddavg = 0, ddmin = 1e30;

#pragma omp parallel for default(shared) reduction(+:ddavg) reduction(min:ddmin)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		scalar rowdd[bs];
		for(int i = 0; i < bs; i++)
			rowdd[i] = 0;

		const index diagp = mat.diagind[irow];

		// add the off-diagonal entries of the diagonal block
		for(int i = 0; i < bs; i++)
			for(int j = 0; j < bs; j++)
				if(i != j)
					rowdd[i] += std::abs(data[diagp](i,j));

		// other off-diagonal entries
		for(index jj = mat.diagind[irow]+1; jj < mat.browendptr[irow]; jj++)
		{
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					rowdd[i] += std::abs(data[jj](i,j));
		}

		for(int i = 0; i < bs; i++)
			rowdd[i] = 1.0 - rowdd[i]/std::abs(data[diagp](i,i));

		for(int i = 0; i < bs; i++)
		{
			if(ddmin > rowdd[i])
				ddmin = rowdd[i];
			ddavg += rowdd[i];
		}
	}

	return {ddavg/(mat.nbrows*bs), ddmin};
}

template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,2> diagonal_dominance_lower(const SRMatrixStorage<const scalar,const index>&& mat)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);

	scalar ddavg = 0, ddmin = 1e30;

#pragma omp parallel for default(shared) reduction(+:ddavg) reduction(min:ddmin)
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		scalar rowdd[bs];
		for(int i = 0; i < bs; i++)
			rowdd[i] = 0;

		for(index jj = mat.browptr[irow]; jj < mat.diagind[irow]; jj++)
		{
			for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					rowdd[i] += std::abs(data[jj](i,j));
		}

		for(int i = 0; i < bs; i++)
			rowdd[i] = 1.0 - rowdd[i];   //< Unit block lower triangular matrix

		for(int i = 0; i < bs; i++)
		{
			if(ddmin > rowdd[i])
				ddmin = rowdd[i];
			ddavg += rowdd[i];
		}
	}

	return {ddavg/(mat.nbrows*bs), ddmin};
}

}
