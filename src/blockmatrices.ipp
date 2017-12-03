/** \file blockmatrices.ipp
 * \brief Implementation of block matrix methods,
 *   including the specialized set of methods for block size 1.
 * \author Aditya Kashi
 * \date 2017-08
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

#include "blockmatrices.hpp"
#include <Eigen/LU>

namespace blasted {

/// Shorthand for dependent templates for Eigen segment function for vectors
#define SEG template segment
/// Shorthand for dependent templates for Eigen block function for matrices
#define BLK template block

/// Matrix-vector product for BSR matrices
/** The template parameter Mattype is the type of the Eigen Matrix that the array of non-zero entries
 * of the matrix should be mapped to.
 * The reason this is a template parameter is that this type changes depending on whether
 * the array of non-zero entries ConstRawBSRMatrix::vals is arragned row-major or column-major within
 * each block. If it's row-major, the type should be Eigen::Matrix<scalar,Dynamic,bs,RowMajor> but
 * if it's column-major, Mattype should be Eigen::Matrix<scalar,bs,Dynamic,ColMajor>.
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_matrix_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const xx,
		scalar *const __restrict yy)
{
	Eigen::Map<const Vector<scalar>> x(xx, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> y(yy, mat->nbrows*bs);

	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");

	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		y.SEG<bs>(irow*bs) = Vector<scalar>::Zero(bs);

		// loop over non-zero blocks of this block-row
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = mat->bcolind[jj];
			if(Mattype::IsRowMajor)	
				y.SEG<bs>(irow*bs).noalias() 
					+= a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
			else
				y.SEG<bs>(irow*bs).noalias() 
					+= a * data.BLK<bs,bs>(0,jj*bs) * x.SEG<bs>(jcol*bs);
		}
	}
}

/// Computes z := a Ax + by for  scalars a and b and vectors x and y
/** 
 * \param[in] mat The BSR matrix
 * \warning xx must not alias zz.
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_gemv3(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz)
{
	Eigen::Map<const Vector<scalar>> x(xx, mat->nbrows*bs);
	Eigen::Map<const Vector<scalar>> y(yy, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, mat->nbrows*bs);

	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		z.SEG<bs>(irow*bs) = b * y.SEG<bs>(irow*bs);

		// loop over non-zero blocks of this block-row
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			const index jcol = mat->bcolind[jj];
			if(Mattype::IsRowMajor)	
				z.SEG<bs>(irow*bs).noalias() += 
					a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
			else
				z.SEG<bs>(irow*bs).noalias() += 
					a * data.BLK<bs,bs>(0,jj*bs) * x.SEG<bs>(jcol*bs);
		}
	}
}

/// Computes and stores the inverses of diagonal blocks
/** Allocates the storage if necessary.
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_jacobi_setup(const ConstRawBSRMatrix<scalar,index> *const mat,
		scalar *const dblocks 
	)
{
	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat->nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat->nbrows*bs
		);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		if(Mattype::IsRowMajor)
			dblks.BLK<bs,bs>(irow*bs,0) = data.BLK<bs,bs>(mat->diagind[irow]*bs,0).inverse();
		else
			dblks.BLK<bs,bs>(0,irow*bs) = data.BLK<bs,bs>(0,mat->diagind[irow]*bs).inverse();
}

/// Applies the block-Jacobi preconditioner assuming inverses of diagonal blocks have been computed 
template <typename scalar, typename index, int bs, class Mattype>
void block_jacobi_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,
		const scalar *const rr, scalar *const __restrict zz)
{
	Eigen::Map<const Vector<scalar>> r(rr, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, mat->nbrows*bs);
	
	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat->nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat->nbrows*bs
		);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		if(Mattype::IsRowMajor)
			z.SEG<bs>(irow*bs).noalias() = dblks.BLK<bs,bs>(irow*bs,0) * r.SEG<bs>(irow*bs);
		else
			z.SEG<bs>(irow*bs).noalias() = dblks.BLK<bs,bs>(0,irow*bs) * r.SEG<bs>(irow*bs);
}

/// Computes inverses of diagonal blocks and allocates and zeros a temporary storage vector
/** \param[in] mat The BSR matrix
 * \param[in,out] dblocks An array to hold the inverse of diagonal blocks
 * \param[out] ytemp A temporary vector needed for SGS; allocated and zeroed here
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_sgs_setup(const ConstRawBSRMatrix<scalar,index> *const mat,
		scalar *const dblocks,
		Vector<scalar>& ytemp
	)
{
	block_jacobi_setup<scalar,index,bs,Mattype>(mat, dblocks);

	ytemp.resize(mat->nbrows*bs,1);
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows*bs; i++)
	{
		ytemp.data()[i] = 0;
	}
}

/// Applies the block SGS preconditioner
/** If multiple threads are used, the iteration is asynchronous \cite async:anzt_triangular .
 * Assumes inverses of diagonal blocks have been computed and stored, 
 * and that \ref ytemp, a temporary storage space, has been allocated.
 * \param[in] mat The BSR matrix
 * \param[in] dblocks An array holding the inverse of diagonal blocks
 * \param ytemp A pre-allocated temporary vector needed for SGS
 * \param[in] napplysweeps Number of sweeps to use for asynchronous block-SGS application
 * \param[in] thread_chunk_size Batch size of allocation of work-items to thread contexts
 * \param[in] rr The input vector to apply the preconditioner to
 * \param[in,out] zz The output vector
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_sgs_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,
		Vector<scalar>& ytemp,
		const int napplysweeps,
		const int thread_chunk_size,
		const scalar *const rr, scalar *const __restrict zz)
{
	Eigen::Map<const Vector<scalar>> r(rr, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, mat->nbrows*bs);

	static_assert(std::is_same<Mattype, Matrix<scalar,Dynamic,bs,RowMajor>>::value 
			|| std::is_same<Mattype, Matrix<scalar,bs,Dynamic,ColMajor>>::value,
		"Invalid matrix type!");
	
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<const Mattype> dblks(dblocks, 
			Mattype::IsRowMajor ? mat->nbrows*bs : bs,
			Mattype::IsRowMajor ? bs : mat->nbrows*bs
		);

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

			for(index jj = mat->browptr[irow]; jj < mat->diagind[irow]; jj++)
				if(Mattype::IsRowMajor)
					inter += data.BLK<bs,bs>(jj*bs,0)*ytemp.SEG<bs>(mat->bcolind[jj]*bs);
				else
					inter += data.BLK<bs,bs>(0,jj*bs)*ytemp.SEG<bs>(mat->bcolind[jj]*bs);

			if(Mattype::IsRowMajor)
				ytemp.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(irow*bs,0) 
			                                          * (r.SEG<bs>(irow*bs) - inter);
			else
				ytemp.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(0,irow*bs) 
			                                          * (r.SEG<bs>(irow*bs) - inter);
		}
	}

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = mat->nbrows-1; irow >= 0; irow--)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
			
			// compute U z
			for(index jj = mat->diagind[irow]+1; jj < mat->browptr[irow+1]; jj++)
				if(Mattype::IsRowMajor)
					inter += data.BLK<bs,bs>(jj*bs,0) * z.SEG<bs>(mat->bcolind[jj]*bs);
				else
					inter += data.BLK<bs,bs>(0,jj*bs) * z.SEG<bs>(mat->bcolind[jj]*bs);

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			if(Mattype::IsRowMajor)
				z.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(irow*bs,0) 
					* ( data.BLK<bs,bs>(mat->diagind[irow]*bs,0)*ytemp.SEG<bs>(irow*bs) - inter );
			else
				z.SEG<bs>(irow*bs) = dblks.BLK<bs,bs>(0,irow*bs) 
					* ( data.BLK<bs,bs>(0,mat->diagind[irow]*bs)*ytemp.SEG<bs>(irow*bs) - inter );
		}
	}
}

/// Search through inner indices
/** Finds the position in the index arary that has value indtofind
 * Searches between positions
 * \param[in] start, and
 * \param[in] end
 */
template <typename index>
static inline void inner_search(const index *const aind, 
		const index start, const index end, 
		const index indtofind, index *const pos)
{
	for(index j = start; j < end; j++) {
		if(aind[j] == indtofind) {
			*pos = j;
			break;
		}
	}
}

/// Constructs the block-ILU0 factorization using a block variant of the Chow-Patel procedure
/// \cite ilu:chowpatel
/** There is currently no pre-scaling of the original matrix A, unlike the scalar ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 *
 * \param[in] mat The BSR matrix
 * \param[in] nbuildsweeps Number of asynchronous sweeps to use for parallel builds
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[out] iluvals The ILU factorization non-zeros, accessed using the block-row pointers, 
 *   block-column indices and diagonal pointers of the original BSR matrix
 * \param[out] ytemp A temporary vector, needed for applying the ILU0 factors, allocated here
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_ilu0_setup(const ConstRawBSRMatrix<scalar,index> *const mat,
		const int nbuildsweeps, const int thread_chunk_size,
		scalar *const ilu, Vector<scalar>& ytemp
	)
{
	Eigen::Map<const Mattype> data(mat->vals, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);
	Eigen::Map<Mattype> iluvals(ilu, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
			{
				if(irow > mat->bcolind[j])
				{
					Matrix<scalar,bs,bs> sum = data.BLK<bs,bs>(j*bs,0);

					for(index k = mat->browptr[irow]; 
					    (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
					    k++  ) 
					{
						index pos = -1;
						inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

						if(pos == -1) continue;

						sum.noalias() -= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
					}

					iluvals.BLK<bs,bs>(j*bs,0).noalias()
						= sum * iluvals.BLK<bs,bs>(mat->diagind[mat->bcolind[j]]*bs,0).inverse();
				}
				else
				{
					// compute u_ij
					iluvals.BLK<bs,bs>(j*bs,0) = data.BLK<bs,bs>(j*bs,0);

					for(index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index mat->bcolind[j],
						 * between the diagonal index of row mat->bcolind[k] 
						 * and the last index of row bcolind[k]
						 */
						inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

						if(pos == -1) continue;

						iluvals.BLK<bs,bs>(j*bs,0).noalias()
							-= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
					}
				}
			}
		}
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		iluvals.BLK<bs,bs>(mat->diagind[irow]*bs,0) 
			= iluvals.BLK<bs,bs>(mat->diagind[irow]*bs,0).inverse().eval();
}

/// Applies the block-ILU0 factorization using a block variant of the asynch triangular solve in
/// \cite async:anzt_triangular
/**
 * \param[in] mat The BSR matrix
 * \param[in] iluvals The ILU factorization non-zeros, accessed using the block-row pointers, 
 *   block-column indices and diagonal pointers of the original BSR matrix
 * \param ytemp A pre-allocated temporary vector, needed for applying the ILU0 factors
 * \param[in] napplysweeps Number of asynchronous sweeps to use for parallel application
 * \param[in] thread_chunk_size The number of work-items to assign to thread-contexts in one batch
 *   for dynamically scheduled threads - should not be too small or too large
 * \param[in] r The RHS vector of the preconditioning problem Mz = r
 * \param[in,out] z The solution vector of the preconditioning problem Mz = r
 */
template <typename scalar, typename index, int bs, class Mattype>
inline
void block_ilu0_apply( const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const ilu,
		Vector<scalar>& ytemp,
		const int napplysweeps, const int thread_chunk_size,
		const scalar *const r, 
        scalar *const __restrict z
	)
{
	Eigen::Map<const Vector<scalar>> ra(r, mat->nbrows*bs);
	Eigen::Map<Vector<scalar>> za(z, mat->nbrows*bs);
	Eigen::Map<const Mattype> iluvals(ilu, 
			Mattype::IsRowMajor ? mat->browptr[mat->nbrows]*bs : bs,
			Mattype::IsRowMajor ? bs : mat->browptr[mat->nbrows]*bs
		);

	// No scaling like z := Sr done here
	
	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < mat->nbrows; i++)
		{
			Matrix<scalar,bs,1> sum = Matrix<scalar,bs,1>::Zero();

			for(index j = mat->browptr[i]; j < mat->diagind[i]; j++)
				sum += iluvals.BLK<bs,bs>(j*bs,0) * ytemp.SEG<bs>(mat->bcolind[j]*bs);
			
			ytemp.SEG<bs>(i*bs) = ra.SEG<bs>(i*bs) - sum;
		}
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			Matrix<scalar,bs,1> sum = Matrix<scalar,bs,1>::Zero();

			for(index j = mat->diagind[i]+1; j < mat->browptr[i+1]; j++)
				sum += iluvals.BLK<bs,bs>(j*bs,0) * za.SEG<bs>(mat->bcolind[j]*bs);
			
			za.SEG<bs>(i*bs) = 
				iluvals.BLK<bs,bs>(mat->diagind[i]*bs,0) * (ytemp.SEG<bs>(i*bs) - sum);
		}
	}

	// No correction of z needed because no scaling
}


template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(BSR), 
	  mat{nullptr, nullptr, nullptr, nullptr, 0},
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{
	std::cout << "BSRMatrix: Initialized matrix without allocation, with\n    "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(BSR), owner{true}, 
	mat{nullptr, nullptr, nullptr, nullptr, n_brows},
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{
	constexpr int bs2 = bs*bs;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]*bs2];
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) 
	{
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	std::cout << "BSRMatrix: Setup with matrix with " << mat.nbrows << " block-rows,\n    "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows, index *const brptrs,
		index *const bcinds, scalar *const values, index *const diaginds,
		const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(BSR), owner{false},
	  mat{brptrs, bcinds, values, diaginds, n_brows},
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{ }

template <typename scalar, typename index, int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	if(owner) {
		delete [] mat.vals;
		delete [] mat.bcolind;
		delete [] mat.browptr;
		delete [] mat.diagind;
	}
	mat.bcolind = mat.browptr = mat.diagind = nullptr; mat.vals = nullptr;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setStructure(const index n_brows,
		const index *const bcinds, const index *const brptrs)
{
	delete [] mat.vals;
	delete [] mat.bcolind;
	delete [] mat.browptr;
	delete [] mat.diagind;

	mat.nbrows = n_brows;
	constexpr int bs2 = bs*bs;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]*bs2];
	owner = true;
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) 
	{
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	std::cout << "BSRMatrix:  Allocated storage for matrix with " << mat.nbrows << " block-rows.\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setAllZero()
{
	const index nnz = mat.browptr[mat.nbrows]*bs*bs;
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < nnz; i++)
		mat.vals[i] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setDiagZero()
{
	constexpr int bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
#pragma omp simd
		for(index jj = mat.diagind[irow]*bs2; jj < (mat.diagind[irow]+1)*bs2; jj++)
			mat.vals[jj] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2) 
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = mat.browptr[startr]; j < mat.browptr[startr+1]; j++) {
		if(mat.bcolind[j] == startc) 
		{
			for(int k = 0; k < bs2; k++)
				mat.vals[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index param)
{
	constexpr int bs2 = bs*bs;
	const index startr = starti/bs;
	const index pos = mat.diagind[startr];
	for(int k = 0; k < bs2; k++)
#pragma omp atomic update
		mat.vals[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2)
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;

	for(index j = mat.browptr[startr]; j < mat.browptr[startr+1]; j++) {
		if(mat.bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				mat.vals[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::scaleAll(const scalar factor)
{
#pragma omp parallel for default(shared)
	for(index iz = 0; iz < mat.browptr[mat.nbrows]; iz++)
	{
#pragma omp simd
		for(index k = 0; k < bs*bs; k++)
			mat.vals[iz*bs*bs + k] *= factor;
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	block_matrix_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), 
			a, xx, yy);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	block_gemv3<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), 
			a, xx, b, yy, zz);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiSetup()
{
	block_jacobi_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
		(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), dblocks.data());
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
	block_jacobi_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
		( reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), dblocks.data(), 
			rr, zz);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precSGSSetup()
{
	block_sgs_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>
		(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), dblocks.data(),ytemp);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	block_sgs_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), dblocks.data(), ytemp,
			napplysweeps,thread_chunk_size,
			rr, zz);
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUSetup()
{
	if(iluvals.size() < mat.browptr[mat.nbrows]*bs*bs)
	{
#if DEBUG==1
		std::printf(" BSRMatrix: precILUSetup(): First-time setup\n");
#endif

		// Allocate lu
		iluvals.resize(mat.browptr[mat.nbrows]*bs,bs);
#pragma omp parallel for simd default(shared)
		for(index j = 0; j < mat.browptr[mat.nbrows]*bs*bs; j++) {
			iluvals.data()[j] = mat.vals[j];
		}

		// intermediate array for the solve part
		if(ytemp.size() < mat.nbrows*bs) {
			ytemp.resize(mat.nbrows*bs);
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows*bs; i++)
			{
				ytemp.data()[i] = 0;
			}
		}
		else
			std::cout << "! BSRMatrix: precILUSetup(): Temp vector is already allocated!\n";
	}

	block_ilu0_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
		reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		nbuildsweeps, thread_chunk_size, iluvals.data(), ytemp);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUApply(const scalar *const r, 
                                              scalar *const __restrict z) const
{
	block_ilu0_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
		reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		iluvals.data(), ytemp, napplysweeps, thread_chunk_size, 
		r, z);
}

/*template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::printDiagnostic(const char choice) const
{
	std::ofstream fout("blockmatrix.txt");
	for(index i = 0; i < mat.nbrows*bs; i++)
	{
		for(index j = 0; j < mat.nbrows*bs; j++)
		{
			index brow = i/bs; index bcol = j/bs;
			int lcol = j%bs;
			bool found = false;
			for(index jj = mat.browptr[brow]; jj < mat.browptr[brow+1]; jj++)
				if(mat.bcolind[jj] == bcol)
				{
					//fout << " " << mat.vals[jj + lrow*bs + lcol];
					fout << " " << mat.bcolind[jj]*bs+lcol+1;
					found = true;
					break;
				}
			if(!found)
				fout << " 0";
		}
		fout << '\n';
	}
	fout.close();
}*/

////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar, typename index>
inline
void matrix_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const xx, scalar *const __restrict yy) 
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		yy[irow] = 0;

		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			yy[irow] += a * mat->vals[jj] * xx[mat->bcolind[jj]];
		}
	}
}

template <typename scalar, typename index>
inline
void scalar_gemv3(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz)
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		zz[irow] = b * yy[irow];

		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
		{
			zz[irow] += a * mat->vals[jj] * xx[mat->bcolind[jj]];
		}
	}
}

/// Inverts diagonal entries
/** \param[in] mat The matrix
 * \param[in,out] dblocks It must be pre-allocated; contains inverse of diagonal entries on exit
 */
template <typename scalar, typename index>
inline
void scalar_jacobi_setup(const ConstRawBSRMatrix<scalar,index> *const mat,
		scalar *const dblocks)
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		dblocks[irow] = 1.0/mat->vals[mat->diagind[irow]];
}

template <typename scalar, typename index>
inline
void scalar_jacobi_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks,	
		const scalar *const rr, scalar *const __restrict zz)
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < mat->nbrows; irow++)
		zz[irow] = dblocks[irow] * rr[irow];
}

/// Computes inverses of diagonal entries and zeros a temporary storage vector
/** \param[in] mat The matrix
 * \param[in,out] dblocks It must be pre-allocated; contains inverse of diagonal entries on exit
 * \param[in,out] ytemp It must be pre-allocated; zeroed here for use later
 */
template <typename scalar, typename index>
inline
void scalar_sgs_setup(const ConstRawBSRMatrix<scalar,index> *const mat, 
		scalar *const dblocks, scalar *const ytemp)
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
	{
		dblocks[i] = 1.0/mat->vals[mat->diagind[i]];
		ytemp[i] = 0;
	}
}

template <typename scalar, typename index>
inline
void scalar_sgs_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const dblocks, scalar *const __restrict ytemp,
		const int napplysweeps, const int thread_chunk_size,
		const scalar *const rr, scalar *const __restrict zz) 
{
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			scalar inter = 0;

			for(index jj = mat->browptr[irow]; jj < mat->diagind[irow]; jj++)
				inter += mat->vals[jj]*ytemp[mat->bcolind[jj]];

			ytemp[irow] = dblocks[irow] * (rr[irow] - inter);
		}
	}

	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = mat->nbrows-1; irow >= 0; irow--)
		{
			scalar inter = 0;
			
			// compute U z
			for(index jj = mat->diagind[irow]+1; jj < mat->browptr[irow+1]; jj++)
				inter += mat->vals[jj] * zz[mat->bcolind[jj]];

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			zz[irow] = dblocks[irow] * ( mat->vals[mat->diagind[irow]]*ytemp[irow] - inter );
		}
	}
}

/// Computes the scalar ILU0 factorization using asynch iterations \cite ilu:chowpatel
/** \param[in] mat The preconditioner as a CSR matrix
 * \param[in] nbuildweeps The number of asynch sweeps to use for a parallel build
 * \param[in] thread_chunk_size The batch size of allocation of work-items to threads
 * \param[in,out] iluvals A pre-allocated array for storage of the ILU0 factorization
 * \param[in,out] scale A pre-allocated array for storage of diagonal scaling factors
 */
template <typename scalar, typename index>
inline
void scalar_ilu0_setup(const ConstRawBSRMatrix<scalar,index> *const mat,
		const int nbuildsweeps, const int thread_chunk_size,
		scalar *const __restrict iluvals, scalar *const __restrict scale)
{
	// get the diagonal scaling matrix
	
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++)
		scale[i] = 1.0/std::sqrt(mat->vals[mat->diagind[i]]);

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(int isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < mat->nbrows; irow++)
		{
			for(index j = mat->browptr[irow]; j < mat->browptr[irow+1]; j++)
			{
				if(irow > mat->bcolind[j])
				{
					scalar sum = scale[irow] * mat->vals[j] * scale[mat->bcolind[j]];

					for(index k = mat->browptr[irow]; 
					    (k < mat->browptr[irow+1]) && (mat->bcolind[k] < mat->bcolind[j]); 
					    k++  ) 
					{
						index pos = -1;
						inner_search<index> ( mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos );

						if(pos == -1) {
							continue;
						}

						sum -= iluvals[k]*iluvals[pos];
					}

					iluvals[j] = sum / iluvals[mat->diagind[mat->bcolind[j]]];
				}
				else
				{
					// compute u_ij
					iluvals[j] = scale[irow]*mat->vals[j]*scale[mat->bcolind[j]];

					for(index k = mat->browptr[irow]; 
							(k < mat->browptr[irow+1]) && (mat->bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index mat->bcolind[j], 
						 * between the diagonal index of row mat->bcolind[k] 
						 * and the last index of row mat->bcolind[k]
						 */
						inner_search(mat->bcolind, mat->diagind[mat->bcolind[k]], 
								mat->browptr[mat->bcolind[k]+1], mat->bcolind[j], &pos);

						if(pos == -1) continue;

						iluvals[j] -= iluvals[k]*iluvals[pos];
					}
				}
			}
		}
	}
}

template <typename scalar, typename index>
inline
void scalar_ilu0_apply(const ConstRawBSRMatrix<scalar,index> *const mat,
		const scalar *const iluvals, const scalar *const scale,
		scalar *const __restrict ytemp,
		const int napplysweeps, const int thread_chunk_size,
		const scalar *const ra, scalar *const __restrict za) 
{
	// initially, z := Sr
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat->nbrows; i++) {
		za[i] = scale[i]*ra[i];
	}
	
	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < mat->nbrows; i++)
		{
			scalar sum = 0;
			for(index j = mat->browptr[i]; j < mat->diagind[i]; j++)
			{
				sum += iluvals[j] * ytemp[mat->bcolind[j]];
			}
			ytemp[i] = za[i] - sum;
		}
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(int isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = mat->nbrows-1; i >= 0; i--)
		{
			scalar sum = 0;
			for(index j = mat->diagind[i]+1; j < mat->browptr[i+1]; j++)
			{
				sum += iluvals[j] * za[mat->bcolind[j]];
			}
			za[i] = 1.0/iluvals[mat->diagind[i]] * (ytemp[i] - sum);
		}
	}

	// correct z
#pragma omp parallel for simd default(shared)
	for(int i = 0; i < mat->nbrows; i++)
		za[i] = za[i]*scale[i];
}

template <typename scalar, typename index>
inline
BSRMatrix<scalar,index,1>::BSRMatrix(const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(CSR), mat{nullptr,nullptr,nullptr,nullptr,0},
	dblocks{nullptr}, iluvals{nullptr}, scale{nullptr}, ytemp{nullptr},
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{
	std::cout << "BSRMatrix<1>: Initialized CSR matrix with "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index>
inline
BSRMatrix<scalar,index,1>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(CSR), owner{true},
	mat{nullptr,nullptr,nullptr,nullptr,n_brows},
	dblocks(nullptr), iluvals(nullptr), scale(nullptr), ytemp(nullptr),
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]];
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) {
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	
	std::cout << "BSRMatrix<1>: Set up CSR matrix with " << mat.nbrows << " rows,\n    "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(const index nrows, index *const brptrs,
		index *const bcinds, scalar *const values, index *const diaginds,
		const int n_buildsweeps, const int n_applysweeps)
	: AbstractMatrix<scalar,index>(CSR), owner{false},
	mat{brptrs,bcinds,values,diaginds,nrows},
	dblocks(nullptr), iluvals(nullptr), scale(nullptr), ytemp(nullptr),
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{ }

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::~BSRMatrix()
{
	if(owner) {
		delete [] mat.vals;
		delete [] mat.bcolind;
		delete [] mat.browptr;
		delete [] mat.diagind;
	}

	delete [] dblocks;
	delete [] iluvals;
	delete [] ytemp;
	delete [] scale;

	mat.bcolind = mat.browptr = mat.diagind = nullptr;
	mat.vals = dblocks = iluvals = ytemp = scale = nullptr;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setStructure(const index n_brows,
		const index *const bcinds, const index *const brptrs)
{
	delete [] mat.vals;
	delete [] mat.bcolind;
	delete [] mat.browptr;
	delete [] mat.diagind;

	mat.nbrows = n_brows;
	mat.browptr = new index[mat.nbrows+1];
	mat.bcolind = new index[brptrs[mat.nbrows]];
	mat.diagind = new index[mat.nbrows];
	mat.vals = new scalar[brptrs[mat.nbrows]];
	owner = true;
	for(index i = 0; i < mat.nbrows+1; i++)
		mat.browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[mat.nbrows]; i++)
		mat.bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < mat.nbrows; irow++) {
		for(index j = mat.browptr[irow]; j < mat.browptr[irow+1]; j++)
			if(mat.bcolind[j] == irow) {
				mat.diagind[irow] = j;
				break;
			}
	}
	
	std::cout << "BSRMatrix<1>: Set up CSR matrix with " << mat.nbrows << " rows.\n";
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setAllZero()
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < mat.browptr[mat.nbrows]; i++)
		mat.vals[i] = 0;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setDiagZero()
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < mat.nbrows; irow++)
		mat.vals[mat.diagind[irow]] = 0;
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			if(mat.bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(mat.bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: submitBlock: Invalid block!!\n";
#endif
			mat.vals[jj] = buffer[locrow*bsj+k];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index bs)
{
	// update the block, row-wise
	for(index irow = starti; irow < starti+bs; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.diagind[irow]-locrow; jj < mat.diagind[irow]-locrow+bs; jj++)
		{
			index loccol = jj-mat.diagind[irow]+locrow;
#pragma omp atomic update
			mat.vals[jj] += buffer[locrow*bs+loccol];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
		{
			if(mat.bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(mat.bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: updateBlock: Invalid block!!\n";
#endif
#pragma omp atomic update
			mat.vals[jj] += buffer[locrow*bsi+k];
			k++;
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::scaleAll(const scalar factor)
{
#pragma omp parallel for simd default(shared)
	for(index iz = 0; iz < mat.browptr[mat.nbrows]; iz++)
	{
		mat.vals[iz] *= factor;
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	matrix_apply(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), a, xx, yy);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	scalar_gemv3(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), a, xx, b, yy, zz);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precJacobiSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.browptr[mat.nbrows]];
		std::cout << " CSR Matrix: precJacobiSetup(): Allocating.\n";
	}

	scalar_jacobi_setup(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat), dblocks);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
	scalar_jacobi_apply(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		dblocks, rr, zz);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precSGSSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.browptr[mat.nbrows]];
		delete [] ytemp;
		ytemp = new scalar[mat.nbrows];
		std::cout << " CSR Matrix: precSGSSetup(): Allocating.\n";
	}
	
	scalar_sgs_setup(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		dblocks, ytemp);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	scalar_sgs_apply(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		dblocks, ytemp, napplysweeps, thread_chunk_size,
		rr, zz);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precILUSetup()
{
	if(!iluvals)
	{
#ifdef DEBUG
		std::printf(" CSR Matrix: precILUSetup(): First-time setup\n");
#endif

		// Allocate lu
		iluvals = new scalar[mat.browptr[mat.nbrows]];
#pragma omp parallel for simd default(shared)
		for(int j = 0; j < mat.browptr[mat.nbrows]; j++) {
			iluvals[j] = mat.vals[j];
		}

		// intermediate array for the solve part; NOT ZEROED
		if(!ytemp) {
			ytemp = new scalar[mat.nbrows];
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows; i++)
			{
				ytemp[i] = 0;
			}
		}
		else
			std::cout << "! BSRMatrix<1>: precILUSetup(): Temp vector is already allocated!\n";
		
		if(!scale)
			scale = new scalar[mat.nbrows];	
		else
			std::cout << "! BSRMatrix<1>: precILUSetup(): Scale vector is already allocated!\n";
	}

	scalar_ilu0_setup(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		nbuildsweeps, thread_chunk_size, iluvals, scale);
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precILUApply(const scalar *const __restrict ra, 
                                              scalar *const __restrict za) const
{
	scalar_ilu0_apply(reinterpret_cast<const ConstRawBSRMatrix<scalar,index>*>(&mat),
		iluvals, scale, ytemp, napplysweeps, thread_chunk_size,
		ra, za);
}

/*template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::printDiagnostic(const char choice) const
{
	std::ofstream fout("pointmatrix.txt");
	for(index i = 0; i < mat.nbrows; i++)
	{
		for(index j = 0; j < mat.nbrows; j++)
		{
			bool found = false;
			for(index jj = mat.browptr[i]; jj < mat.browptr[i+1]; jj++)
				if(mat.bcolind[jj] == j)
				{
					fout << " " << mat.bcolind[jj]+1;
					found = true;
					break;
				}
			if(!found)
				fout << " 0";
		}
		fout << '\n';
	}
	fout.close();
}*/

////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar, typename index, int bs, StorageOptions stor>
BSRMatrixView<scalar,index,bs,stor>::BSRMatrixView(const index n_brows, const index *const brptrs,
		const index *const bcinds, const scalar *const values, const index *const diaginds,
		const int n_buildsweeps, const int n_applysweeps)
	: MatrixView<scalar,index>(BSR),
	  mat{brptrs, bcinds, values, diaginds, n_brows}, dblocks{nullptr}, iluvals{nullptr},
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{ }

template <typename scalar, typename index, int bs, StorageOptions stor>
BSRMatrixView<scalar, index, bs,stor>::~BSRMatrixView()
{
	delete [] dblocks;
	delete [] iluvals;
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	if(stor == RowMajor)
		block_matrix_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(&mat, a, xx, yy);
	else
		block_matrix_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(&mat, a, xx, yy);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::gemv3(const scalar a, const scalar *const __restrict xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	if(stor == RowMajor)
		block_gemv3<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(&mat, a, xx, b, yy, zz);
	else
		block_gemv3<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(&mat, a, xx, b, yy, zz);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precJacobiSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.nbrows*bs*bs];
#if DEBUG==1
		std::cout << " precJacobiSetup(): Allocating.\n";
#endif
	}
	
	if(stor == RowMajor)
		block_jacobi_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(&mat, dblocks);
	else
		block_jacobi_setup<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(&mat, dblocks);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
	if(stor == RowMajor)
		block_jacobi_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>( &mat, dblocks, rr, zz);
	else
		block_jacobi_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>( &mat, dblocks, rr, zz);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precSGSSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.nbrows*bs*bs];
#if DEBUG==1
		std::cout << " precSGSSetup(): Allocating.\n";
#endif
	}
	
	if(stor == RowMajor)
		block_sgs_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(&mat, dblocks, ytemp);
	else
		block_sgs_setup<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(&mat, dblocks, ytemp);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	if(stor == RowMajor)
		block_sgs_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			&mat, dblocks, ytemp, napplysweeps,thread_chunk_size, rr, zz);
	else
		block_sgs_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(
			&mat, dblocks, ytemp, napplysweeps,thread_chunk_size, rr, zz);
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precILUSetup()
{
	if(!iluvals)
	{
#if DEBUG==1
		std::printf(" BSRMatrixView: precILUSetup(): First-time setup\n");
#endif

		// Allocate lu
		iluvals = new scalar[mat.browptr[mat.nbrows]*bs*bs];
#pragma omp parallel for simd default(shared)
		for(index j = 0; j < mat.browptr[mat.nbrows]*bs*bs; j++) {
			iluvals[j] = mat.vals[j];
		}

		// intermediate array for the solve part
		if(ytemp.size() < mat.nbrows*bs) {
			ytemp.resize(mat.nbrows*bs);
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows*bs; i++)
			{
				ytemp.data()[i] = 0;
			}
		}
		else
			std::cout << "! BSRMatrix: precILUSetup(): Temp vector is already allocated!\n";
	}

	if(stor == RowMajor)
		block_ilu0_setup<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			&mat, nbuildsweeps, thread_chunk_size, iluvals, ytemp);
	else
		block_ilu0_setup<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(
			&mat, nbuildsweeps, thread_chunk_size, iluvals, ytemp);
}

template <typename scalar, typename index, int bs, StorageOptions stor>
void BSRMatrixView<scalar,index,bs,stor>::precILUApply(const scalar *const r, 
                                              scalar *const __restrict z) const
{
	if(stor == RowMajor)
		block_ilu0_apply<scalar,index,bs,Matrix<scalar,Dynamic,bs,RowMajor>>(
			&mat, iluvals, ytemp, napplysweeps, thread_chunk_size, r, z);
	else
		block_ilu0_apply<scalar,index,bs,Matrix<scalar,bs,Dynamic,ColMajor>>(
			&mat, iluvals, ytemp, napplysweeps, thread_chunk_size, r, z);
}


template <typename scalar, typename index>
CSRMatrixView<scalar,index>::CSRMatrixView(const index nrows, const index *const brptrs,
		const index *const bcinds, const scalar *const values, const index *const diaginds,
		const int n_buildsweeps, const int n_applysweeps)
	: MatrixView<scalar,index>(CSR),
	mat{brptrs,bcinds,values,diaginds,nrows},
	dblocks(nullptr), iluvals(nullptr), scale(nullptr), ytemp(nullptr),
	nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{ }

template <typename scalar, typename index>
CSRMatrixView<scalar,index>::~CSRMatrixView()
{ 
	delete [] dblocks;
	delete [] iluvals;
	delete [] scale;
	delete [] ytemp;
	dblocks = iluvals = scale = ytemp = nullptr;
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	matrix_apply(&mat, a, xx, yy);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	scalar_gemv3(&mat, a, xx, b, yy, zz);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precJacobiSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.nbrows];
		std::cout << " CSR MatrixView: precJacobiSetup(): Initial setup.\n";
	}

	scalar_jacobi_setup(&mat, dblocks);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
	scalar_jacobi_apply(&mat, dblocks, rr, zz);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precSGSSetup()
{
	if(!dblocks) {
		dblocks = new scalar[mat.nbrows];
		delete [] ytemp;
		ytemp = new scalar[mat.nbrows];
		std::cout << " CSR MatrixView: precSGSSetup(): Initial setup.\n";
	}
	
	scalar_sgs_setup(&mat, dblocks, ytemp);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	scalar_sgs_apply(&mat, dblocks, ytemp, napplysweeps, thread_chunk_size, rr, zz);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precILUSetup()
{
	if(!iluvals)
	{
		std::printf(" CSR MatrixView: precILUSetup(): First-time setup\n");

		// Allocate lu
		iluvals = new scalar[mat.browptr[mat.nbrows]];
#pragma omp parallel for simd default(shared)
		for(int j = 0; j < mat.browptr[mat.nbrows]; j++) {
			iluvals[j] = mat.vals[j];
		}

		// intermediate array for the solve part; NOT ZEROED
		if(!ytemp) {
			ytemp = new scalar[mat.nbrows];
#pragma omp parallel for simd default(shared)
			for(index i = 0; i < mat.nbrows; i++)
			{
				ytemp[i] = 0;
			}
		}
		else
			std::cout << "! BSRMatrixView<1>: precILUSetup(): Temp vector is already allocated!\n";
		
		if(!scale)
			scale = new scalar[mat.nbrows];	
		else
			std::cout << "! BSRMatrixView<1>: precILUSetup(): Scale vector is already allocated!\n";
	}

	scalar_ilu0_setup(&mat, nbuildsweeps, thread_chunk_size, iluvals, scale);
}

template <typename scalar, typename index>
void CSRMatrixView<scalar,index>::precILUApply(const scalar *const __restrict ra, 
                                              scalar *const __restrict za) const
{
	scalar_ilu0_apply(&mat, iluvals, scale, ytemp, napplysweeps, thread_chunk_size, ra, za);
}

}

