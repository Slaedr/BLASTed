/** \file
 * \brief Declares the types of solver operations available
 * \author Aditya Kashi
 */

#ifndef BLASTED_SOLVERTYPES_H
#define BLASTED_SOLVERTYPES_H

#ifdef __cplusplus
extern "C" {
#endif

	/// The types of solver operations that BLASTed provides
	typedef enum {BLASTED_JACOBI,
	              BLASTED_GS,
	              BLASTED_SGS,
	              BLASTED_ILU0,
	              BLASTED_SAPILU0,
	              BLASTED_NO_PREC,
	              BLASTED_EXTERNAL
	} BlastedSolverType;

#ifdef __cplusplus
}
#endif

#endif
