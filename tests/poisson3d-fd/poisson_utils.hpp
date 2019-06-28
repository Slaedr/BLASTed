/** \file
 * \brief Declaration of functionality associated with Poisson test cases
 */

#ifndef BLASTED_TEST_POISSON_UTILS_H
#define BLASTED_TEST_POISSON_UTILS_H

#include "cartmesh.hpp"

namespace test {

/// Specification of mesh (read from an input file)
struct MeshSpec {
	PetscInt npdim[3];
	PetscReal rmin[3];
	PetscReal rmax[3];
	char gridtype[50];
};

MeshSpec readInputFile(const MPI_Comm comm, const char *confile);

/// Build a Cartesian mesh from the mesh spec
CartMesh createMesh(MPI_Comm comm, const MeshSpec spec);

}
#endif
