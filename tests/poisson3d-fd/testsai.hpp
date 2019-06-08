/** \file
 * \brief Some tests related to sparse approximate inverses
 */

#ifndef BLASTED_TESTS_SAI_H
#define BLASTED_TESTS_SAI_H

#include <petscmat.h>
#include "cartmesh.hpp"

int test_sai(const bool fullsai, const CartMesh& m, const Mat A);

#endif
