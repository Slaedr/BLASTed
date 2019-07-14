# test executables

add_library(solvers solvers.cpp)
target_link_libraries(solvers solverops blockmatrices)
set_property(TARGET solvers PROPERTY POSITION_INDEPENDENT_CODE ON)
if(CXX_COMPILER_CLANG)
  target_compile_options(solvers PRIVATE "-Wno-error=pass-failed")
endif()

add_executable(testsolve runsolvetest.cpp testsolve.cpp)
target_link_libraries(testsolve solvers solverops coomatrix )
set_property(TARGET testsolve PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(testreorderedsolve testreorderedsolve.cpp)
target_link_libraries(testreorderedsolve solvers solverops coomatrix )
set_property(TARGET testreorderedsolve PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(testunstructsaipattern testunstructuredsaipattern.cpp)
target_link_libraries(testunstructsaipattern solverops rawmatrixutils)

# Tests

add_subdirectory(helper_algs)
add_subdirectory(mat_ops)
add_subdirectory(perftest)

set(TCS 300)

# valgrind --leak-check=full

add_test(NAME SAIAndIncompleteSAIPatternsUnstructured COMMAND testunstructsaipattern)

add_test(NAME SPDCSRJacobi COMMAND ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi init_none init_none csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
  1e-10 1e-9 200  ${TCS}
)

add_test(NAME SPDCSRSGS COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs init_zero init_zero csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
  1e-10 1e-10 200 ${TCS}
)

add_test(NAME SPDCSRILU0 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 init_zero init_zero csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
  1e-10 1e-10 200 ${TCS}
)

add_test(NAME CSRJacobi COMMAND ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi init_none init_none csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200  ${TCS}
)
add_test(NAME CSRSGS COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs init_zero init_zero csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)
add_test(NAME CSRILU0 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 init_zero init_zero csr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

add_test(NAME BSR4JacobiRowmajor COMMAND ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi init_zero init_zero bsr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

add_test(NAME BSR4SGSRowmajor COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs init_zero init_zero bsr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

add_test(NAME BSR4ILU0Rowmajor COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 init_zero init_zero bsr rowmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
  )

add_test(NAME BSR4NoneColmajor COMMAND ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs none init_zero init_zero bsr colmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-11 1e-8 1000 ${TCS}
)

add_test(NAME BSR4JacobiColmajor COMMAND ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi init_zero init_zero bsr colmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

add_test(NAME BSR4SGSColmajor COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs init_zero init_zero bsr colmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

add_test(NAME BSR4ILU0Colmajor COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
  ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 init_zero init_zero bsr colmajor
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
  1e-10 1e-8 200 ${TCS}
)

# add_test(NAME Reorder_CSRILU0_msc00726 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
#   ${CMAKE_CURRENT_BINARY_DIR}/testreorderedsolve bcgs ilu0 init_original init_zero csr rowmajor
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
#   1e-10 1e-8 200 ${TCS}
#   )
# add_test(NAME Reorder_CSRILU0 COMMAND env OMP_NUM_THREADS=1 ${SEQEXEC} ${SEQTASKS}
#   ${CMAKE_CURRENT_BINARY_DIR}/testreorderedsolve bcgs ilu0 init_original init_zero csr rowmajor
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
#   ${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
#   1e-10 1e-8 200 ${TCS}
#   )

if(WITH_PETSC)
  add_library(petsctestutils testutils.cpp testutils.c)
  target_link_libraries(petsctestutils blasted_petsc utils)
  set_property(TARGET petsctestutils PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_executable(testpetscsolver runpetsctest.c)
  target_link_libraries(testpetscsolver coomatrix blasted_petsc petsctestutils)

  #add_test(NAME SerialPetsc-BSR4-Jacobi COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 1
  #	${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_jacobi.perc
  #	)
  add_test(NAME MPIPetsc-BSR4-Jacobi 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_jacobi.perc
	-error_tolerance 2.3e-13 -test_type issame
	)

  add_test(NAME MPIPetsc-BSR4-SGS 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type baij -test_type issame
	-error_tolerance 1e-13
	)

  add_test(NAME MPIPetsc-BSR4-ILU0 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type baij
	-error_tolerance 1e-13 -test_type issame
	)

  add_test(NAME SerialPetsc-CSR-ILU0 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type aij -test_type issame
	-error_tolerance 1e-13
	)

  add_test(NAME MPIPetsc-BSR4-Jacobi-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type jacobi -test_type issame
	-error_tolerance 1e-13
	)
  add_test(NAME MPIPetsc-BSR4-GS-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type sor -ref_sub_pc_sor_forward -blasted_pc_type gs -test_type issame
	-error_tolerance 1e-13
	)
  add_test(NAME MPIPetsc-BSR4-SGS-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type sor -ref_sub_pc_sor_local_symmetric -blasted_pc_type sgs -test_type issame
	-error_tolerance 1e-13
	)

  add_test(NAME ThreadedPetsc-BSR4-Level-SGS
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type baij
	-blasted_pc_type level_sgs -test_type issame -error_tolerance 1e-13
	)

  add_test(NAME ThreadedPetsc-BSR4-Async-Level-ILU0
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type baij
	-blasted_pc_type async_level_ilu0 -blasted_async_sweeps 5,1 -test_type issame
	-error_tolerance 1e-7
	)

  ## Our scalar SGS preconditioner is not the same as PETSc's, because the latter does block
  ##  preconditioning even for aij matrices.
  add_test(NAME MPIPetsc-CSR-SGS 
  COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
  	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type aij -test_type convergence
  	)
  ## Fails! Our scalar SGS relaxation is not the same as PETSc's.
  #add_test(NAME MPIPetsc-CSR-SGS-Relaxation 
  # COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type aij
  #	-ref_sub_pc_type sor -ref_sub_pc_sor_local_symmetric -blasted_pc_type sgs
  #	)

  add_test(NAME ThreadedRelaxation-Petsc-BSR4-Async-GS-ResidualConvergence 
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type gs -test_type convergence
	)
  add_test(NAME ThreadedRelaxation-Petsc-BSR4-Async-SGS-ItersLowerBound 
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type sgs -test_type upper_bound_its
	)

  add_test(NAME ThreadedPreconditioning-Petsc-BSR4-Async-SGS
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncpreconditioning.perc -mat_type baij
	-ref_sub_pc_type sor -blasted_pc_type sgs -blasted_async_sweeps 1,4
	-test_type compare_its -error_tolerance 0.01
	)

  add_test(NAME ThreadedPreconditioning-Petsc-BSR4-Async-ILU0
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncpreconditioning.perc -mat_type baij
	-ref_sub_pc_type ilu -blasted_pc_type ilu0 -blasted_async_sweeps 2,3
	-test_type compare_its -error_tolerance 0.01
	)
  
  #add_test(NAME MPIPetscMKL-BSR4-SGS 
  # COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs_mkl.perc
  #)
  #add_test(NAME MPIPetscMKL-BSR4-ILU0 
  # COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0_mkl.perc
  #)
  
endif()

add_subdirectory(poisson3d-fd)
