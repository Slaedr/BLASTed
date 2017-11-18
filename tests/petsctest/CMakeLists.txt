# test executables
if(WITH_PETSC)
	add_executable(testcsr_poisson poisson3d.cpp cartmesh.cpp)
	target_link_libraries(testcsr_poisson blasted_petsc ${MPI_CXX})
	target_include_directories(testcsr_poisson PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()

# Tests

