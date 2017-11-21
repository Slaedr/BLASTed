# test executables
if(WITH_PETSC)
	add_executable(testcsr_poisson poisson3d.cpp poisson3d_fd.cpp cartmesh.cpp)
	target_link_libraries(testcsr_poisson blasted_petsc)

# Tests
add_test(NAME PoissonCSRPetscILU0 COMMAND mpirun -n 1 ./testcsr_poisson ${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson_csr_petsc.perc)

endif()


