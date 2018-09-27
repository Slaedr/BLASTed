/** \file
 * \brief Declaration of types of initialization for asynchronous preconditioners
 * \author Aditya Kashi
 */

#ifndef BLASTED_ASYNC_INITIALIZATION_TYPES
#define BLASTED_ASYNC_INITIALIZATION_TYPES

namespace blasted {

/// Types of initialization (initial guess) for asynchronous ILU factorization
enum FactInit {
	/// Initializate the factorization with zeros
	INIT_F_ZERO,
	/// Initializate the factorization with the original matrix (of which we need the factorization)
	INIT_F_ORIGINAL,
	/// Initialize the factorization such that async. ILU(0) factorization gives async. SGS at worst
	INIT_F_SGS,
	/// No initial value
	INIT_F_NONE
};

/// Types of initialization (initial guess) for asynchronous triangular solves
enum ApplyInit {
	/// Initializate the factorization with zeros
	INIT_A_ZERO,
	/// Initialize in a way that the triangular solve is Jacobi at worst
	INIT_A_JACOBI,
	/// No initial value
	INIT_A_NONE
};

}

#endif
