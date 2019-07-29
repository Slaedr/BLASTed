/** \file
 * \brief Implementation of computation of some properties of local matrices
 */

#include "matrix_properties.hpp"

namespace blasted {

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
		for(int i = 0; i < bs; i++)
			for(int j = i+1; j < bs; j++)
				rowdd[i] += std::abs(data[diagp](i,j));

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

template std::array<double,2>
diagonal_dominance_upper<double,int,1,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,2>
diagonal_dominance_upper<double,int,4,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,2>
diagonal_dominance_upper<double,int,5,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);

template std::array<double,2>
diagonal_dominance_upper<double,int,4,RowMajor>(const SRMatrixStorage<const double,const int>&& mat);

template <typename scalar, typename index, int bs, StorageOptions stor>
std::array<scalar,2> diagonal_dominance_lower(const SRMatrixStorage<const scalar,const index>&& mat)
{
	using Blk = Block_t<scalar,bs,stor>;
	const Blk *data = reinterpret_cast<const Blk*>(&mat.vals[0]);

	scalar ddavg = 0, ddmin = 1e30;

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

template std::array<double,2>
diagonal_dominance_lower<double,int,1,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,2>
diagonal_dominance_lower<double,int,4,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);
template std::array<double,2>
diagonal_dominance_lower<double,int,5,ColMajor>(const SRMatrixStorage<const double,const int>&& mat);

template std::array<double,2>
diagonal_dominance_lower<double,int,4,RowMajor>(const SRMatrixStorage<const double,const int>&& mat);

}
