/** \file solverops_base.hpp
 * \brief Header for (local) thread-parallel preconditioners and iterations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_BASE_H
#define BLASTED_SOLVEROPS_BASE_H

#include <limits>

#include "linearoperator.hpp"
#include "srmatrixdefs.hpp"

namespace blasted {

/// Application parameters for certain operators - usually relaxations
template <typename scalar>
struct SolveParams {
	scalar rtol;        ///< Relative tolerance
	scalar atol;        ///< Absolute  tolerance
	scalar dtol;        ///< tolerance for divergence (on the relative tolerance)
	bool ctol;          ///< Whether to check for the tolerances
	int maxits;         ///< Maximum iterations
};
	
/// Generic 'preconditioner' interface
/** We use "preconditioner" for want of a better term. It is used here in the general sense of
 * a single iteration of any linear iterative solver.
 */
template<typename scalar, typename index>
class Preconditioner : public AbstractLinearOperator<scalar,index>
{
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");

public:
	Preconditioner(const StorageType storagetype);

	virtual ~Preconditioner();

	/// Returns the dimension (number of rows) of the operator
	virtual index dim() const = 0;

	/// Compute the preconditioner
	virtual void compute() = 0;

	/// Set parameters that may be used by subclasses
	void setApplyParams(const SolveParams<scalar> sparams) { solveparams = sparams; }

	/// To apply the preconditioner
	virtual void apply(const scalar *const x, scalar *const __restrict y) const = 0;

protected:

	/// Optional apply parameters \sa SolveParams
	SolveParams<scalar> solveparams;
};

/// Preconditioners that operate on sparse row matrices
template<typename scalar, typename index>
class SRPreconditioner : public Preconditioner<scalar,index>
{
public:
	SRPreconditioner();
	
	/// Wraps a sparse-row matrix described by 4 arrays, and recomputes the preconditioner
	/** Calls \ref Preconditioner::compute 
	 * \param[in] n_brows Number of (block-)rows
	 * \param[in] brptrs Array of (block-)row pointers
	 * \param[in] bcinds Array of (block-)column indices
	 * \param[in] values Non-zero values
	 * \param[in] dinds Array of pointers to diagonal entries/blocks
	 *
	 * Does not take ownership of the 4 arrays; they are not cleaned up in the destructor either.
	 */
	void wrap(const index n_brows, const index *const brptrs,
	          const index *const bcinds, const scalar *const values, const index *const dinds);
	
	/// Compute the preconditioner
	/** Does not usually need to be called by the user because it is automatically invoked whenever
	 * the matrix is changed via \ref wrap.
	 */
	virtual void compute() = 0;

protected:
	/// Kind of storage that is used
	StorageType _type;

	/// Matrix wrapper
	CRawBSRMatrix<scalar,index> mat;
};

/// Identity operator as preconditioner 
template<typename scalar, typename index>
class NoPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	/// Set the dimension of the preconditioning operator
	NoPreconditioner(const index n_dim);

	/// Returns the dimension (number of rows) of the operator
	index dim() const { return ndim; }

	/// Does nothing
	void compute() { }
	
	/// Does nothing but copy the input argument into the output argument
	void apply(const scalar *const x, scalar *const __restrict y) const;

protected:
	index ndim;
};

}

#endif
