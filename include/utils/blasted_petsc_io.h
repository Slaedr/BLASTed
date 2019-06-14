/** \file blasted_petsc_io.h
 * \brief I/O operations involving PETSc objects
 * \author Aditya Kashi
 */

#ifndef BLASTED_PETSC_IO
#define BLASTED_PETSC_IO

#include <petscmat.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Sets a Vec from a file
/** \warning Call only in a single-process run!
 * \param[in] a If not NULL, the structure of the new vec v is a duplipate of
 *    the structure of a.
 */
PetscErrorCode readVecFromFile(const char *const file, MPI_Comm comm, Mat a, Vec *const v);

/// Sets a Mat from a file containing a sparse matrix in coordinate format
/** \warning Call only in a single-process run!
 */
PetscErrorCode readMatFromCOOFile(const char *const file, MPI_Comm comm, Mat *const A);

#ifdef __cplusplus
}
#endif

#endif
