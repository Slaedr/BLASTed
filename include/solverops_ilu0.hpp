/** \file solverops_ilu0.hpp
 * \brief Header for (local) thread-parallel incomplete LU preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_ILU0_H
#define BLASTED_SOLVEROPS_ILU0_H

#include "solverops_base.hpp"
#include "reorderingscaling.hpp"
#include "async_initialization_decl.hpp"
#include "ilu_pattern.hpp"

namespace blasted {

/// Asynchronous block-ILU(0) preconditioner for sparse-row matrices
/** Finds \f$ \tilde{L} \f$ and \f$ \tilde{U} \f$ such that
 * \f[ \tilde{L} \tilde{U} = A \f]
 * is approximately satisifed, with \f$ \tilde{L} \f$ block unit lower triangular.
 * The sparsity if A is preserved, so this is ILU(0).
 * This block version is adapted from the scalar version in \cite ilu:chowpatel.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
class AsyncBlockILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");

public:
	/** \param matrix An approximation of the linear system matrix to compute the preconditioner from
	 * \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param thread_chunk_size Size of thread chunks in dynamically parallel loops
	 * \param fact_inittype Type of initialization to use for factorization
	 * \param apply_inittype Type of initialization to use for application
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 * \param compute_remainder Whether to compute diagnostics about the solve (has non-trivial cost)
	 * \param jacobi_iter Whether to do nonlinear Jacobi iterations instead of asynchronous iterations
	 *             for factorization
	 */
	AsyncBlockILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
	                                const int nbuildsweeps, const int napplysweeps,
	                                const bool use_scaling, const int thread_chunk_size,
	                                const FactInit fact_inittype, const ApplyInit apply_inittype,
	                                const bool threadedfactor=true, const bool threadedapply=true,
	                                const bool compute_remainder = false, const bool jacobi_iter = false);

	~AsyncBlockILU0_SRPreconditioner();

	/// Returns the number of rows of the operator
	index dim() const { return mat.nbrows*bs; }

	bool relaxationAvailable() const { return false; }

	/// Compute the preconditioner \sa block_ilu0_setup
	/** We assume that sucessive calls to this function maintain the same sparsity pattern of the
	 * original matrix. ILU location precomputations are not repeated.
	 */
	PrecInfo compute();

	/// Applies a block LU factorization L U z = r
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;

	using Blk = Block_t<scalar,bs,stor>;
	using Seg = Segment_t<scalar,bs>;

	/// Precomputed positions in \ref iluvals to help with factorization
	ILUPositions<index> plist;

	/// Storage for L and U factors
	/** Use \ref bcolind and \ref browptr to access the storage,
	 * as the non-zero structure of this matrix is same as the original matrix.
	 */
	scalar *iluvals;

	/// Matrix used to scale the original matrix before factorization
	scalar *scale;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool usescaling;                       ///< Whether to scale the matrix before ILU
	const bool threadedfactor;         ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;          ///< True for thread-parallel LU application
	const int nbuildsweeps;
	const int napplysweeps;
	const int thread_chunk_size;
	const FactInit factinittype;
	const ApplyInit applyinittype;
	const bool compute_remainder;
	const bool jacobiiter;                       ///< Whether to do Jacobi iters instead of asynchronous

	void setup_storage();
};

/// Asynchronous scalar ILU(0) operator for sparse-row matrices
template <typename scalar, typename index>
class AsyncILU0_SRPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	/** \param nbuildsweeps Number of asynchronous sweeps used to compute the LU factors
	 * \param napplysweeps Number of asynchronous sweeps used to apply the preconditioner
	 * \param thread_chunk_size Size of thread chunks in dynamically parallel loops
	 * \param use_scaling Whether to scale the matrix symmetrically before factorization
	 * \param fact_inittype Type of initialization to use for factorization
	 * \param apply_inittype Type of initialization to use for application
	 * \param threadedfactor If false, the preconditioner is computed sequentially
	 * \param threadedapply If false, the preconditioner is applied sequentially
	 */
	AsyncILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
	                           const int nbuildsweeps, const int napplysweeps,
	                           const bool use_scaling, const int thread_chunk_size,
	                           const FactInit fact_inittype, const ApplyInit apply_inittype,
	                           const bool compute_preconditioner_info,
	                           const bool threadedfactor=true, const bool threadedapply=true,
	                           const bool jacobi_iter = false);

	virtual ~AsyncILU0_SRPreconditioner();

	/// Returns the number of rows
	index dim() const { return mat.nbrows; }

	bool relaxationAvailable() const { return false; }

	/// Compute the preconditioner
	PrecInfo compute();

	/// To apply the preconditioner
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;

	/// Precomputed positions in \ref iluvals to help with factorization
	ILUPositions<index> plist;

	/// Storage for L and U factors
	scalar *iluvals;

	/// Matrix used to scale the original matrix before factorization
	scalar *scale;

	/// Temporary storage for result of application of L
	scalar *ytemp;

	const bool usescaling;                       ///< Whether to scale the matrix before ILU
	const bool threadedfactor;                   ///< True for thread-parallel ILU0 factorization
	const bool threadedapply;                    ///< True for thread-parallel LU application
	const int nbuildsweeps;                      ///< Number of async sweeps for building ILU factors
	const int napplysweeps;                      ///< Number of async sweeps for applying ILU factors

	/// Number of work-items in each dynamic job assigned to a thread
	const int thread_chunk_size;

	const FactInit factinittype;
	const ApplyInit applyinittype;
	const bool compute_precinfo;                 ///< Whether to compute expensive quantities for analysis
	const bool jacobiiter;                       ///< Whether to do Jacobi iters instead of asynchronous

	/// Allocates memory for storing LU factors and initializes it, as well as temporary data
	/** \param scaling Set to true to allocate storage for the scaling vector that's applied to
	 * the matrix A before computing the ILU factors.
	 */
	void setup_storage();
};

/// EXPERIMENTAL -
/// Asynchronous scalar ILU(0) that uses an external (re-)ordering (and scaling) before factorization
/** The reordering and scaling are updated every time the preconditioner is computed.
 * The scaling is as in AsyncILU0_SRPreconditioner - symmetric scaling to make diagonal entries 1.
 * Note that this does not accept a 'use_scaling' boolean parameter in the constructor.
 */
template <typename scalar, typename index>
class ReorderedAsyncILU0_SRPreconditioner : public AsyncILU0_SRPreconditioner<scalar,index>
{
public:
	/** \see AsyncILU0_SRPreconditioner
	 * \param reorderscale The reordering and scaling object use at every iteration
	 */
	ReorderedAsyncILU0_SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix,
	                                    ReorderingScaling<scalar,index,1> *const reorderscale,
	                                    const int nbuildsweeps, const int napplysweeps,
	                                    const int thread_chunk_size,
	                                    const FactInit fact_init_type, const ApplyInit apply_init_type,
	                                    const bool threadedfactor=true, const bool threadedapply=true);

	~ReorderedAsyncILU0_SRPreconditioner();

	/// Apply the ordering and scaling and then compute the preconditioner
	PrecInfo compute();

	/// Apply the preconditioner and apply ordering and scaling to the output
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing but throw an exception
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using SRPreconditioner<scalar,index>::pmat;
	using AsyncILU0_SRPreconditioner<scalar,index>::plist;
	using AsyncILU0_SRPreconditioner<scalar,index>::iluvals;
	using AsyncILU0_SRPreconditioner<scalar,index>::scale;
	using AsyncILU0_SRPreconditioner<scalar,index>::ytemp;
	using AsyncILU0_SRPreconditioner<scalar,index>::threadedfactor;
	using AsyncILU0_SRPreconditioner<scalar,index>::threadedapply;
	using AsyncILU0_SRPreconditioner<scalar,index>::nbuildsweeps;
	using AsyncILU0_SRPreconditioner<scalar,index>::napplysweeps;
	using AsyncILU0_SRPreconditioner<scalar,index>::thread_chunk_size;
	using AsyncILU0_SRPreconditioner<scalar,index>::factinittype;
	using AsyncILU0_SRPreconditioner<scalar,index>::applyinittype;
	using AsyncILU0_SRPreconditioner<scalar,index>::jacobiiter;
	using AsyncILU0_SRPreconditioner<scalar,index>::setup_storage;

	/// Reordered and scaled form of the original preconditioning matrix
	RawBSRMatrix<scalar,index> rsmat;

	/// Computes a reordering and a scaling, in this case, whenever the matrix \ref mat is changed
	ReorderingScaling<scalar,index,1> *const reord;
};

#ifdef HAVE_MC64

/// EXPERIMENTAL - 
/// Asynchronous scalar ILU(0) that uses one of the MC64 (\cite mc64_manual) orderings
/** The reordering is updated every time the preconditioner is computed. Therefore, this is
 * very inefficient.
 */
template <typename scalar, typename index>
class MC64_AsyncILU0_SRPreconditioner : public ReorderedAsyncILU0_SRPreconditioner<scalar,index>
{
public:
	/** Sets options
	 * \param job The MC64 job ID. Must be 1,2,3,4 or 5. See \cite mc64_manual for details.
	 * \warning For job 5, we don't apply the scaling. However, for all jobs, symmetric scaling is
	 *   applied after reordering.
	 * \sa AsyncILU0_SRPreconditioner
	 */
	MC64_AsyncILU0_SRPreconditioner(const int job, const int nbuildsweeps, const int napplysweeps,
	                                const int thread_chunk_size,
	                                const FactInit fact_init_type, const ApplyInit apply_init_type,
	                                const bool threadedfactor=true, const bool threadedapply=true);

	~MC64_AsyncILU0_SRPreconditioner();

	//using ReorderedAsyncILU0_SRPreconditioner<scalar,index>::compute;

protected:
	using SRPreconditioner<scalar,index>::mat;
	using AsyncILU0_SRPreconditioner<scalar,index>::plist;
	using AsyncILU0_SRPreconditioner<scalar,index>::iluvals;
	using AsyncILU0_SRPreconditioner<scalar,index>::scale;
	using AsyncILU0_SRPreconditioner<scalar,index>::ytemp;
	using AsyncILU0_SRPreconditioner<scalar,index>::threadedfactor;
	using AsyncILU0_SRPreconditioner<scalar,index>::threadedapply;
	using AsyncILU0_SRPreconditioner<scalar,index>::nbuildsweeps;
	using AsyncILU0_SRPreconditioner<scalar,index>::napplysweeps;
	using AsyncILU0_SRPreconditioner<scalar,index>::thread_chunk_size;
	using AsyncILU0_SRPreconditioner<scalar,index>::factinittype;
	using AsyncILU0_SRPreconditioner<scalar,index>::applyinittype;
	using AsyncILU0_SRPreconditioner<scalar,index>::setup_storage;
	using ReorderedAsyncILU0_SRPreconditioner<scalar,index>::rsmat;
	using ReorderedAsyncILU0_SRPreconditioner<scalar,index>::reord;

	/// MC64 job index
	const int job;
};

#endif

} // end namespace

#endif
