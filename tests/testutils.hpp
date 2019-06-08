/** \file
 * \brief Some utility functions to assist testing
 */

#ifndef BLASTED_TESTS_UTILITIES_H
#define BLASTED_TESTS_UTILITIES_H

#include <petscmat.h>
#include "srmatrixdefs.hpp"

/// Convert error code to exception
void petsc_check(const int ierr);

/// Wrap a local PETSc matrix
/** Do not destroy the returned raw matrix! It's managed by PETSc.
 */
CRawBSRMatrix<PetscScalar,PetscInt> wrapLocalPetscMat(Mat A, const int bs);

#endif
