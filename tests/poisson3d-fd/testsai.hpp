/** \file
 * \brief Some tests related to sparse approximate inverses
 */

#ifndef BLASTED_TESTS_SAI_H
#define BLASTED_TESTS_SAI_H

#include <petscmat.h>
#include "cartmesh.hpp"

inline PetscInt getMatRowIdx(const CartMesh& m, const PetscInt testpoint[3])
{
	return (testpoint[2]-1)*(m.gnpoind(0)-2)*(m.gnpoind(1)-2)
		+ (testpoint[1]-1)*(m.gnpoind(0)-2) + testpoint[0]-1;
}

#endif
