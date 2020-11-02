/** \file
 * \brief Declares the types of solver operations available
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVERTYPES_H
#define BLASTED_SOLVERTYPES_H

#ifdef __cplusplus
extern "C" {
#endif

	/// The types of solver iterations that BLASTed provides
	typedef enum {BLASTED_JACOBI,
	              BLASTED_GS,
	              BLASTED_SGS,
	              BLASTED_ILU0,
	              BLASTED_SEQILU0,
	              BLASTED_SFILU0,
	              BLASTED_SAPILU0,
	              BLASTED_JAC_ILU0,
	              BLASTED_CSC_BGS,
	              BLASTED_LEVEL_SGS,
	              BLASTED_ASYNC_LEVEL_ILU0,
	              BLASTED_NO_PREC,
	              BLASTED_EXTERNAL
	} BlastedSolverType;

	typedef enum {
	               BLASTED_ITER_JACOBI,
	               BLASTED_ITER_GAUSS_SEIDEL,
	               BLASTED_ITER_ASYNC
	} BlastedIterType;


/// Integer that can be used for sweep counts to request a sequential operation
#define BLASTED_SEQUENTIAL_SYMBOL -1

#ifdef __cplusplus
}
#endif

#endif
