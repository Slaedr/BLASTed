# test executables
if(WITH_PETSC)
  add_library(poisson poisson_setup.cpp poisson3d_fd.cpp cartmesh.cpp)
  target_link_libraries(poisson petsctestutils)
  set_property(TARGET poisson PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_executable(testcsr_poisson poisson_test.c)
  target_link_libraries(testcsr_poisson poisson)

  add_executable(testsai testsai.cpp)
  target_link_libraries(testsai poisson solverops)

  add_executable(testtriangularsai testtriangularsai.cpp)
  target_link_libraries(testtriangularsai poisson solverops rawmatrixutils)

  # Tests

  add_test(NAME SAIPatternPoisson
	COMMAND testsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control fullsai
	)
  add_test(NAME IncompleteSAIPatternPoisson
	COMMAND testsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control incomplete
	)
  add_test(NAME FullSAIPatternPoissonUpperTriangular
  	COMMAND testtriangularsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control fullsai upper
  	)
  add_test(NAME FullSAIPatternPoissonLowerTriangular
  	COMMAND testtriangularsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control fullsai lower
  	)
  add_test(NAME IncompleteSAIPatternPoissonLowerTriangular
  	COMMAND testtriangularsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control incomplete lower
  	)
  add_test(NAME IncompleteSAIPatternPoissonUpperTriangular
  	COMMAND testtriangularsai ${CMAKE_CURRENT_SOURCE_DIR}/input/small_poisson.control incomplete upper
  	)

  add_test(NAME MPIPoissonCSRPetscJacobiRelaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type jacobi -blasted_pc_type jacobi -test_type issame
	)

  add_test(NAME MPIPoissonCSRPetsc-GS-Relaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -ref_sub_pc_sor_forward -blasted_pc_type gs -test_type issame
	)

  add_test(NAME MPIPoissonCSRPetsc-SGS-Relaxation
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -blasted_pc_type sgs -test_type issame
	)

  add_test(NAME MPIThreadedPoissonCSRPetsc-LevelSGS-Relaxation
	COMMAND ${MPIEXEC} -n 2 ${THREADOPTS} ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_localrelaxation.perc
	-ref_sub_pc_type sor -blasted_pc_type level_sgs -test_type issame
	)

  add_test(NAME SerialPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc
	-blasted_use_symmetric_scaling false -test_type issame)

  add_test(NAME MPIPoissonCSRPetscILU0 COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_ilu0.perc
	-blasted_use_symmetric_scaling false -test_type issame)

  add_test(NAME MPIPoissonCSRPetscSGS COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 4 ./testcsr_poisson 
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson_csr_sgs.perc -test_type issame)

  add_test(NAME ThreadedPoissonCSRPetscSGS COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc -blasted_async_sweeps 1,4
	-ref_sub_pc_type sor -blasted_pc_type sgs -test_type issame -error_tolerance 1e-7)

  add_test(NAME ThreadedPoissonCSRPetscILU0 COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc -blasted_async_sweeps 4,6
	-ref_sub_pc_type ilu -blasted_pc_type ilu0
	-blasted_use_symmetric_scaling false -test_type issame -error_tolerance 1e-6)

  add_test(NAME ThreadedPoissonCSRPetscSAPILU0 COMMAND ${MPIEXEC} -n 1 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc
	-ref_sub_pc_type ilu -blasted_pc_type sapilu0 -blasted_async_sweeps 10,1
	-blasted_use_symmetric_scaling 0
	-test_type issame -error_tolerance 1e-6 -num_runs 3)

  add_test(NAME ThreadedPoissonCSRPetsc-LevelSGS
	COMMAND ${SEQEXEC} ${THREADOPTS} ${CMAKE_CURRENT_BINARY_DIR}/testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc
	-ref_sub_pc_type sor -blasted_pc_type level_sgs -test_type issame)

  # Does not work with hyper-threading, for some reason
  add_test(NAME ThreadedPoissonCSRPetsc-Async-Level-ILU0
	COMMAND ${SEQEXEC} ${THREADOPTS} ${CMAKE_CURRENT_BINARY_DIR}/testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/thread_poisson_csr.perc 
	-ref_sub_pc_type ilu -blasted_pc_type async_level_ilu0 -blasted_async_sweeps 100,1
	-blasted_async_fact_init_type init_original -blasted_use_symmetric_scaling false
	-test_type issame -error_tolerance 1e-6)

  add_test(NAME ThreadedRelaxationPoissonPetsc-Async-GS-ItersLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type gs -test_type upper_bound_its)

  add_test(NAME ThreadedRelaxationPoissonPetsc-Async-SGS-ItersLowerBound
	COMMAND ${MPIEXEC} ${THREADOPTS} -n 1 ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson.control 
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/../input/asyncrelaxation.perc 
	-ref_sub_pc_type jacobi -blasted_pc_type sgs -test_type upper_bound_its)

  add_test(NAME SerialPoissonCSRPetsc-CSC-BGS
	COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ./testcsr_poisson
	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-large.control
	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/mpi_poisson.perc
	-ref_sub_pc_type sor -ref_sub_pc_sor_backward -blasted_pc_type cscbgs
	-test_type issame
	)

  # add_test(NAME AsyncPrecPoissonCSRPetsc-CSC-BGS
  # 	COMMAND ${SEQEXEC} ${THREADOPTS} ./testcsr_poisson_threaded
  # 	${CMAKE_CURRENT_SOURCE_DIR}/input/poisson-large.control
  # 	-options_file  ${CMAKE_CURRENT_SOURCE_DIR}/input/async_poisson.perc
  # 	-ref_sub_pc_type sor -ref_sub_pc_sor_backward -blasted_pc_type cscbgs -num_runs 3
  # 	-test_type compare_its -error_tolerance_factor 0.01
  # 	)

endif()


