/** \file solverops_base.hpp
 * \brief Header for (local) thread-parallel preconditioners and iterations
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVEROPS_BASE_H
#define BLASTED_SOLVEROPS_BASE_H

#include <limits>
#include <vector>
#include <boost/align/aligned_allocator.hpp>

#include "linearoperator.hpp"
#include "srmatrixdefs.hpp"

namespace blasted {

/// An aligned dynamic array
template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T,CACHE_LINE_LEN>>;

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

	/// For querying whether a valid relaxation is available via \ref apply_relax
	virtual bool relaxationAvailable() const = 0;

	/// Compute the preconditioner
	virtual void compute() = 0;

	/// Set parameters that may be used by subclasses
	void setApplyParams(const SolveParams<scalar> sparams) { solveparams = sparams; }

	/// To apply the preconditioner
	virtual void apply(const scalar *const x, scalar *const __restrict y) const = 0;

	/// To apply relaxation
	virtual void apply_relax(const scalar *const x, scalar *const __restrict y) const = 0;

protected:

	/// Optional apply parameters \sa SolveParams
	SolveParams<scalar> solveparams;
};

/// Preconditioners that operate on sparse row matrices
template<typename scalar, typename index>
class SRPreconditioner : public Preconditioner<scalar,index>
{
public:
	SRPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix);

	/// Compute the preconditioner
	virtual void compute() = 0;

protected:
	/// Matrix view
	SRMatrixStorage<const scalar, const index> pmat;
	/// Matrix wrapper
	CRawBSRMatrix<scalar,index> mat;
};

/// Identity operator as preconditioner
template<typename scalar, typename index>
class NoPreconditioner : public SRPreconditioner<scalar,index>
{
public:
	/// Set the dimension of the preconditioning operator
	NoPreconditioner(SRMatrixStorage<const scalar, const index>&& matrix, const index bs);

	/// Returns the dimension (number of rows) of the operator
	index dim() const { return ndim; }

	bool relaxationAvailable() const { return false; }

	/// Does nothing
	void compute() { }

	/// Does nothing but copy the input argument into the output argument
	void apply(const scalar *const x, scalar *const __restrict y) const;

	/// Does nothing
	void apply_relax(const scalar *const x, scalar *const __restrict y) const;

protected:
	using SRPreconditioner<scalar,index>::pmat;

	index ndim;
};

}

#endif
