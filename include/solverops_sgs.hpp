/** \file solverops_sgs.hpp
 * \brief Header for (local) thread-parallel Gauss-Seidel type operations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_SGS_H
#define BLASTED_SOLVEROPS_SGS_H

#include "async_initialization_decl.hpp"
#include "solverops_jacobi.hpp"
#include "scmatrixdefs.hpp"

namespace blasted {

/// Asynchronous block-SGS operator for sparse-row matrices
/** \warning While re-wrapping a different matrix, make sure it's of the same dimension
 * as the previous one.
 * 
 * Applies a block symmetric Gauss-Seidel preconditioner ("LU-SGS")
 * Approximately solves (D+L) D^(-1) (D+U) z = r
 * where D, L and U are the diagonal, upper and lower parts of the matrix respectively,
 * by applying asynchronous Jacobi sweeps.
 * This block version is adapted from the scalar version in \cite async:anzt_triangular
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
class AsyncBlockSGS_SRPreconditioner : public BJacobiSRPreconditioner<scalar,index,bs,stor>
{
public:
	/// Create block SGS preconditioner
	/** \param napplysweeps Number of asynchronous application sweeps
	 * \param apply_inittype Type of initialization to use for temporary and output vectors
	 * \param threadchunksize Number of iterations assigned to a thread at a time
	 */
	AsyncBlockSGS_SRPreconditioner(const int napplysweeps, const ApplyInit apply_inittype,
	                               const int threadchunksize);

	~AsyncBlockSGS_SRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }

	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using Preconditioner<scalar,index>::solveparams;
	using SRPreconditioner<scalar,index>::mat;
	using BJacobiSRPreconditioner<scalar,index,bs,stor>::dblocks;

	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;

	const int napplysweeps;
	const ApplyInit ainit;
	const int thread_chunk_size;
};

/// Asynchronous scalar SGS operator for sparse-row matrices
template <typename scalar, typename index>
class AsyncSGS_SRPreconditioner : public JacobiSRPreconditioner<scalar,index>
{
public:
	/// Create asynchronous scalar SGS preconditioner
	/** \param napplysweeps Number of asynchronous application sweeps
	 * \param apply_inittype Type of initialization to use for temporary and output vectors
	 * \param threadchunksize Number of iterations assigned to a thread at a time
	 */
	AsyncSGS_SRPreconditioner(const int napplysweeps, const ApplyInit apply_inittype,
	                          const int threadchunksize);

	~AsyncSGS_SRPreconditioner();

	/// Returns the number of rows
	index dim() const { return mat.nbrows; }

	/// Compute the preconditioner
	void compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Carry out a relaxation solve
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using Preconditioner<scalar,index>::solveparams;
	using SRPreconditioner<scalar,index>::mat;
	using JacobiSRPreconditioner<scalar,index>::dblocks;
	
	/// Temporary storage for the result of the forward Gauss-Seidel sweep
	mutable scalar *ytemp;

	const int napplysweeps;
	const ApplyInit ainit;
	const int thread_chunk_size;
};

/// Backward Gauss-Seidel preconditioner applied column-wise
/** Works fine in single-threaded mode - it's an exact bacward triangular solve. But,
 * \warning The asynchronous mode seems to be inconsistent - it seems to require more Richardson
 * iterations regardless of the number of sweeps.
 * \todo Investigate consistency of the asynchronous iteration.
 */
template <typename scalar, typename index>
class CSC_BGS_Preconditioner : public JacobiSRPreconditioner<scalar,index>
{
	static_assert(std::is_signed<index>::value, "Signed index type required!");

public:
	CSC_BGS_Preconditioner(const int napplysweeps, const int threadchunksize);
	~CSC_BGS_Preconditioner();

	index dim() const { return mat.nbrows; }

	void compute();
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using Preconditioner<scalar,index>::solveparams;
	using SRPreconditioner<scalar,index>::mat;
	using JacobiSRPreconditioner<scalar,index>::dblocks;
	const int napplysweeps;
	const int thread_chunk_size;

	CRawBSCMatrix<scalar,index> cmat;         ///< Storage for original matrix in CSC format
};

}

#endif
