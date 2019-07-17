/** \file
 * \brief Some pre-processing needed for building SAI-type preconditioners
 */

#ifndef BLASTED_SAI_H
#define BLASTED_SAI_H

#include <vector>
#include <boost/align/aligned_allocator.hpp>
#include "srmatrixdefs.hpp"

namespace blasted {

using boost::alignment::aligned_allocator;

/// The pattern required to assemble small least-squares problems for computing a left SAI preconditioner
/** The sparsity pattern of the SAI is set to that of A.
 */
template <typename index>
struct LeftSAIPattern
{
	/// Locations in the original matrix's bcolind array for each entry of all SAI LHS matrices
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> bpos;
	/// Row indices of entries in the SAI LHS matrix corresponding to \ref bpos
	std::vector<int,aligned_allocator<index,CACHE_LINE_LEN>> browind;
	/// Pointers to the start of every block-column of every SAI LHS matrix
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> bcolptr;
	/// Pointers into \ref bcolptr for the start of the LHS matrix of the least-squares problem
	///   for each block-row of the original matrix
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> sairowptr;

	/// Number of non-zero blocks in each block-row of the approximate inverse
	std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> nVars;
	/// Number of equations in the least-squares problem for each block-row of the approx inverse
	std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> nEqns;

	/// Row indec of central vertex (diagonal location) in each vertex's least-squares LHS matrix
	std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> localCentralRow;
	/// Column index of central vertex (diagonal location) in each vertex's least-squares LHS matrix
	//std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> localCentralCol;
};

/// Compute the assembly pattern for a left sparse approximate inverse of the given matrix
template <typename scalar, typename index>
LeftSAIPattern<index> left_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat);

/// Compute the assembly pattern for a left incomplete sparse approximate inverse of the given matrix
template <typename scalar, typename index>
LeftSAIPattern<index> left_incomplete_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat);

/// Compute the approximate inverse matrix using a given SAI pattern
/** \param mat The original matrix
 * \param sp The SAI/ISAI pattern \ref LeftSAIPattern
 * \param thread_chunk_size Number of work-items per chunk given to a thread at a time
 * \param fullsai Set to true for computing a SAI, false for computing an incomplete SAI
 * \param[out] sai The approximate inverse matrix preallocated to the same size as the original matrix
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void compute_SAI(const CRawBSRMatrix<scalar,index>& mat, const LeftSAIPattern<index>& sp,
                 const int thread_chunk_size, const bool fullsai, RawBSRMatrix<scalar,index>& sai);

}

#endif
