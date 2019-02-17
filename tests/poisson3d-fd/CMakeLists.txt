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

  add_test(NAME MPIThreadedPoissonCSRPetsc-LevelSGS-Relaxation
	COMMAND ${MPIEXEC} -n 2 ${THREADOPTS} ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -blasted_pc_type level_sgs
	)

  add_test(NAME SerialPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc)

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

  add_test(NAME ThreadedPoissonCSRPetsc-LevelSGS
	COMMAND ${SEQEXEC} ${THREADOPTS} ${CMAKE_CURRENT_BINARY_DIR}/testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
	-ref_sub_pc_type sor -blasted_pc_type level_sgs)

  # Does not work with hyper-threading, for some reason
  add_test(NAME ThreadedPoissonCSRPetsc-Async-Level-ILU0
	COMMAND ${SEQEXEC} ${THREADOPTS} ${CMAKE_CURRENT_BINARY_DIR}/testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
	-ref_sub_pc_type ilu -blasted_pc_type async_level_ilu0 -blasted_async_sweeps 500,1
	-blasted_async_fact_init_type init_original -error_tolerance_factor 1e5)

  add_test(NAME ThreadedRelaxationPoissonPetsc-Async-GS-ItersLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type gs -test_type upper_bound_its)

  add_test(NAME ThreadedRelaxationPoissonPetsc-Async-SGS-ItersLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson_threaded
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type sgs -test_type upper_bound_its)

  add_test(NAME SerialPoissonCSRPetsc-CSC-BGS
	COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-large.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson.perc
	-ref_sub_pc_type sor -ref_sub_pc_sor_backward -blasted_pc_type cscbgs
	-error_tolerance_factor 1e2
	)

  # add_test(NAME AsyncPrecPoissonCSRPetsc-CSC-BGS
  # 	COMMAND ${SEQEXEC} ${THREADOPTS} ./testcsr_poisson_threaded
  # 	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-large.control
  # 	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/async_poisson.perc
  # 	-ref_sub_pc_type sor -ref_sub_pc_sor_backward -blasted_pc_type cscbgs -num_runs 3
  # 	-test_type compare_its -error_tolerance_factor 0.01
  # 	)

endif()


