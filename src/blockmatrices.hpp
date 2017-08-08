/** \file blockmatrices.hpp
 * \brief Classes for sparse matrices consisting of dense blocks
 * \author Aditya Kashi
 * \date 2017-07-29
 */

#ifndef __BLOCKMATRICES_H
#define __BLOCKMATRICES_H

// get around dependent templates for Eigen
#define SEG template segment
#define BLK template block

#include "linearoperator.hpp"
#include <iostream>
#include <Eigen/LU>

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// Block sparse row matrix
/** Dense blocks stored in a (block-) row-major storage order.
 */
template <typename scalar, typename index, int bs>
class BSRMatrix : public LinearOperator<scalar, index>
{
protected:

	/// Entries of the matrix
	/** All the blocks are stored contiguously as one big block-column
	 * having as many block-rows as the total number of non-zero blocks
	 */
	scalar* vals;

	/// Has space for non-zero values \ref vals been allocated by this class?
	const bool isAllocVals;
	
	/// Block-column indices of blocks in data
	index* bcolind;
	
	/// Stores indices into bcolind where block-rows start
	/** Has size (number of rows + 1). 
	 * The last entry stores the total number of non-zero blocks.
	 */
	index* browptr;

	/// Number of block-rows
	const index nbrows;
	
	/// Stores indices into bcolind if diagonal blocks
	index* diagind;

	/// Storage for factored or inverted diagonal blocks
	Matrix<scalar,Dynamic,bs,RowMajor> dblocks;

	/// Storage for ILU0 factorization
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	Matrix<scalar,Dynamic,bs,RowMajor> iludata;

	/// Storage for intermediate results in preconditioning operations
	mutable Vector<scalar> ytemp;

	/// Number of sweeps used to build preconditioners
	const short nbuildsweeps;

	/// Number of sweeps used to apply preconditioners
	const short napplysweeps;

	/// Thread chunk size for OpenMP parallelism
	const int thread_chunk_size;

public:
	/// Allocates space for the matrix based on the supplied non-zero structure
	/** \param[in] n_brows Total number of block rows
	 * \param[in] bcinds Column indices, simply copied over into \ref bcolind
	 * \param[in] brptrs Row pointers, simply copied into \ref browptr
	 */
	BSRMatrix(const index n_brows, const index *const bcinds, const index *const brptrs,
	         const short nbuildsweeps, const short napplysweeps);

	/// De-allocates memory
	virtual ~BSRMatrix();

	/// Sets all stored entries of the matrix to zero
	void setAllZero();

	/// Sets diagonal blocks to zero
	void setDiagZero();
	
	/// Insert a block of values into the [matrix](\ref data); not thread-safe
	/** \warning NOT thread safe! The caller is responsible for ensuring that no two threads
	 * write to the same location in \ref data at the same time.
	 */
	void submitBlock(const index starti, const index startj, 
			const scalar *const buffer, const long param1, const long param2);

	/// Update a (contiguous) block of values into the [matrix](\ref data)
	/** This is function is thread-safe: each location that needs to be updated is updated
	 * atomically.
	 */
	void updateBlock(const index starti, const index startj, 
			const scalar *const buffer, const long param1, const long param2);
	
	/// Updates the diagonal block of the specified block-row
	/** This function is thread-safe.
	 * \param[in] starti The block-row whose diagonal block is to be updated
	 * \param[in] buffer The values making up the block to be added
	 */
	void updateDiagBlock(const index starti, const scalar *const buffer);

	/// Computes the matrix vector product of this matrix with one vector-- y := a Ax
	virtual void apply(const scalar a, const scalar *const x, scalar *const __restrict y) const;

	/// Almost the BLAS gemv: computes z := a Ax + by for  scalars a and b
	virtual void gemv3(const scalar a, const scalar *const __restrict x, 
			const scalar b, const scalar *const y,
			scalar *const z) const;

	/// Computes inverse or factorization of diagonal blocks for applying Jacobi preconditioner
	void precJacobiSetup();
	
	/// Applies block-Jacobi preconditioner
	void precJacobiApply(const scalar *const r, scalar *const __restrict z) const;

	/// Allocates storage for a vector \ref ytemp required for both SGS and ILU applications
	void allocTempVector();

	/// Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
	void precSGSApply(const scalar *const r, scalar *const __restrict z) const;

	/// Computes an incomplete block lower-upper factorization
	void precILUSetup();

	/// Applies a block LU factorization
	void precILUApply(const scalar *const r, scalar *const __restrict z) const;
};

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const short n_buildsweeps, const short n_applysweeps)
	: vals(nullptr), isAllocVals(true), nbrows{n_brows},
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{
	constexpr int bs2 = bs*bs;
	browptr = new index[nbrows+1];
	bcolind = new index[brptrs[nbrows]];
	diagind = new index[nbrows];
	vals = new scalar[brptrs[nbrows]*bs2];
	for(index i = 0; i < nbrows+1; i++)
		browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[nbrows]; i++)
		bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < nbrows; irow++) {
		for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			if(bcolind[j] == irow) {
				diagind[irow] = j;
				break;
			}
	}
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	if(isAllocVals)
		delete [] vals;
	if(bcolind)
		delete [] bcolind;
	if(browptr)
		delete [] browptr;
	if(diagind)
		delete [] diagind;
	bcolind = browptr = diagind = nullptr;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setAllZero()
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < browptr[nbrows]; i++)
		vals[i] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setDiagZero()
{
	constexpr int bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
#pragma omp simd
		for(index jj = diagind[irow]*bs2; jj < (diagind[irow]+1)*bs2; jj++)
			vals[jj] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const long param1, const long param2) 
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
				vals[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const scalar *const buffer)
{
	constexpr int bs2 = bs*bs;
	const index startr = starti/bs;
	const index pos = diagind[startr];
	for(int k = 0; k < bs2; k++)
#pragma omp atomic update
		vals[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const long param1, const long param2)
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;

	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				vals[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	Eigen::Map<const Vector<scalar>> x(xx, nbrows*bs);
	Eigen::Map<Vector<scalar>> y(yy, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		y.SEG<bs>(irow*bs) = Vector<scalar>::Zero(bs);

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = bcolind[jj];
			y.SEG<bs>(irow*bs).noalias() 
				+= a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
			
			/*for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					y[irow*bs+i] += a * data[jj*bs2 + i*bs+j] * x[jcol*bs+j];*/
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	Eigen::Map<const Vector<scalar>> x(xx, nbrows*bs);
	Eigen::Map<const Vector<scalar>> y(yy, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		z.SEG<bs>(irow*bs) = b * y.SEG<bs>(irow*bs);

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			const index jcol = bcolind[jj];
			z.SEG<bs>(irow*bs).noalias() += 
				a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiSetup()
{
	if(dblocks.size() <= 0) {
		dblocks.resize(nbrows*bs,bs);
#if DEBUG==1
		std::cout << " BSRMatrix: precJacobiSetup(): Allocating.\n";
#endif
	}
	
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		dblocks.BLK<bs,bs>(irow*bs,0) = data.BLK<bs,bs>(diagind[irow]*bs,0).inverse();
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict__ zz) const
{
	Eigen::Map<const Vector<scalar>> r(rr, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		z.SEG<bs>(irow*bs).noalias() = dblocks.BLK<bs,bs>(irow*bs,0) * r.SEG<bs>(irow*bs);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::allocTempVector()
{
	ytemp.resize(nbrows*bs,1);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict__ zz) const
{
	Eigen::Map<const Vector<scalar>> r(rr, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);
	constexpr int bs2 = bs*bs;

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

			for(index jj = browptr[irow]; jj < diagind[irow]; jj++)
				inter += data.BLK<bs,bs>(jj*bs,0)*ytemp.SEG<bs>(bcolind[jj]);

			ytemp.SEG<bs>(irow) = dblocks.BLK<bs,bs>(irow*bs,0) 
			                                          * (r.SEG<bs>(irow) - inter);
		}
	}

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
			
			// compute U z
			for(index jj = diagind[irow]; jj < browptr[irow+1]; jj++)
				inter += data.BLK<bs,bs>(jj*bs,0) * z.SEG<bs>(bcolind[jj]*bs);

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			z.SEG<bs>(irow*bs) = dblocks.BLK<bs,bs>(irow*bs2,0) 
				* ( data.BLK<bs,bs>(diagind[irow]*bs,0)*ytemp.SEG<bs>(irow*bs) - inter );
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUSetup()
{
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, nbrows, bs);
	// TODO
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUApply(const scalar *const r, 
                                              scalar *const __restrict__ z) const
{
	// TODO
}

}
#endif
