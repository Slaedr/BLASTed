/** \file
 * \brief Some utility functions to assist testing
 */

#ifndef BLASTED_TESTS_UTILITIES_H
#define BLASTED_TESTS_UTILITIES_H

#include <petscmat.h>
#include "srmatrixdefs.hpp"

#define PETSCOPTION_STR_LEN 30

namespace blasted {

/// Wrap a local PETSc matrix
SRMatrixStorage<const PetscScalar,const PetscInt> wrapLocalPetscMat(Mat A, const int bs);

}
#endif
