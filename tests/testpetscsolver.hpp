/** \file testpetscsolver.hpp
 * \brief To test preconditioners with PETSc
 * \author Aditya Kashi
 */

#include <string>
#include <fstream>
#include <petscmat.h>
#include "../src/blockmatrices.hpp"

PetscErrorCode readVecFromFile(std::string file, Vec v);

PetscErrorCode readMatFromFile(std::string file, Mat A);

