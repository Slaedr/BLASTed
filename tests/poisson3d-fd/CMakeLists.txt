# test executables
if(WITH_PETSC)
	add_executable(testcsr_poisson poisson3d.cpp poisson3d_fd.cpp cartmesh.cpp)
	target_link_libraries(testcsr_poisson blasted_petsc)
	
	add_executable(testcsr_poisson_threaded poisson3d-threaded.cpp poisson3d_fd.cpp cartmesh.cpp)
	target_link_libraries(testcsr_poisson_threaded blasted_petsc)

# Tests
add_test(NAME SerialPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ./testcsr_poisson 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson_csr_ilu0.perc)

add_test(NAME SerialPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ./testcsr_poisson 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson_csr_sgs.perc)

add_test(NAME SerialPoissonCSRPetscAMGILU0 COMMAND env OMP_NUM_THREADS=1 ./testcsr_poisson 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson-amg.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson_csr_amg.perc)

add_test(NAME MPIPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 8 ./testcsr_poisson 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/mpi_poisson_csr_ilu0.perc)

add_test(NAME MPIPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 mpirun -n 8 ./testcsr_poisson 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/mpi_poisson_csr_sgs.perc)

add_test(NAME ThreadedPoissonCSRPetscSGS COMMAND mpirun -n 1 ./testcsr_poisson_threaded 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/thread_poisson_csr.perc 
-ref_sub_pc_type sor -blasted_pc_type sgs)

add_test(NAME ThreadedPoissonCSRPetscILU0 COMMAND mpirun -n 1 ./testcsr_poisson_threaded 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/thread_poisson_csr.perc 
-ref_sub_pc_type ilu -blasted_pc_type ilu0)

add_test(NAME ThreadedPoissonCSRPetscSAPILU0 COMMAND mpirun -n 1 ./testcsr_poisson_threaded 
${CMAKE_CURRENT_SOURCE_DIR}/../input/poisson.control ${CMAKE_CURRENT_SOURCE_DIR}/../input/thread_poisson_csr.perc 
-ref_sub_pc_type ilu -blasted_pc_type sapilu0)

endif()


