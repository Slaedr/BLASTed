# test executables
if(WITH_PETSC)
  add_executable(testcsr_poisson poisson3d.cpp poisson3d_fd.cpp cartmesh.cpp)
  target_link_libraries(testcsr_poisson blasted_petsc)
  set_property(TARGET testcsr_poisson PROPERTY POSITION_INDEPENDENT_CODE ON)
  
  add_executable(testcsr_poisson_threaded poisson3d-threaded.cpp poisson3d_fd.cpp cartmesh.cpp)
  target_link_libraries(testcsr_poisson_threaded blasted_petsc)
  set_property(TARGET testcsr_poisson_threaded PROPERTY POSITION_INDEPENDENT_CODE ON)

  # Tests
  #add_test(NAME SerialPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testcsr_poisson 
  #${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
  #-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_ilu0.perc)

  #add_test(NAME SerialPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testcsr_poisson 
  #${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
  #-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_sgs.perc)

  #add_test(NAME SerialPoissonCSRPetscAMGILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1 ./testcsr_poisson 
  #  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-amg.control 
  #  -options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/poisson_csr_amg.perc)

  add_test(NAME MPIPoissonCSRPetscJacobiRelaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type jacobi -blasted_pc_type jacobi
	)

  add_test(NAME MPIPoissonCSRPetsc-GS-Relaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -ref_sub_pc_sor_forward -blasted_pc_type gs
	)

  add_test(NAME MPIPoissonCSRPetsc-SGS-Relaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -blasted_pc_type sgs
	)

  add_test(NAME MPIPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc)

  add_test(NAME MPIPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_sgs.perc)

  add_test(NAME ThreadedPoissonCSRPetscSGS COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson_threaded 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc
	-ref_sub_pc_type sor -blasted_pc_type sgs -test_type compare_error)

  add_test(NAME ThreadedPoissonCSRPetscILU0 COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson_threaded 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc
	-ref_sub_pc_type ilu -blasted_pc_type ilu0 -test_type compare_error -error_tolerance_factor 1e7)

  add_test(NAME ThreadedPoissonCSRPetscSAPILU0 COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson_threaded 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
	-ref_sub_pc_type ilu -blasted_pc_type sapilu0 -test_type compare_error)

  add_test(NAME AsyncRelaxationPoissonPetsc-GS-IterationsLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type gs -test_type compare_its)

  add_test(NAME AsyncRelaxationPoissonPetsc-SGS-IterationsLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type sgs -test_type compare_its)

endif()


