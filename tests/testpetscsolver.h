/** \file testpetscsolver.h
 * \brief To test preconditioners with PETSc
 * \author Aditya Kashi
 */

#include <petscmat.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Sets a Vec from a file
/** The sizes of the Vec, at least the global size, must have been set beforehand.
 */
PetscErrorCode readVecFromFile(const char *const file, Vec v);

/// Sets a Mat from a file containing a sparse matrix in coordinate format
/** MatSetSizes, MatMPIAIJSetPreallocation, MatSeqAIJSetPreallocation,
 * MatMPIBAIJSetPreallocation and MatSeqBAIJSetPreallocation, or MatSetUp, must be called
 * before calling this function.
 */
PetscErrorCode readMatFromCOOFile(const char *const file, Mat A);

#ifdef __cplusplus
}
#endif
