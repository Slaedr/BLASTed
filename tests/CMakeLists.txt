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

# Tests

add_subdirectory(helper_algs)
add_subdirectory(mat_ops)

set(TCS 300)

# valgrind --leak-check=full 

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

if(WITH_PETSC)
  add_executable(testpetscsolver runpetsctest.c)
  target_link_libraries(testpetscsolver coomatrix blasted_petsc)

  add_executable(testthreadedpetscsolver runthreadedpetsctest.c)
  target_link_libraries(testthreadedpetscsolver coomatrix blasted_petsc)
  
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
	-error_tolerance_factor 1e3
	)

  add_test(NAME MPIPetsc-BSR4-SGS 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type baij
	)

  add_test(NAME MPIPetsc-BSR4-ILU0 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type baij
	-error_tolerance_factor 1e3
	)

  add_test(NAME SerialPetsc-CSR-ILU0 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type aij
	)

  add_test(NAME MPIPetsc-BSR4-Jacobi-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type jacobi
	)
  add_test(NAME MPIPetsc-BSR4-GS-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type sor -ref_sub_pc_sor_forward -blasted_pc_type gs
	)
  add_test(NAME MPIPetsc-BSR4-SGS-Relaxation 
	COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} ${MPIOPTS} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type baij
	-ref_sub_pc_type sor -ref_sub_pc_sor_local_symmetric -blasted_pc_type sgs
	)

  ## Fails! Our scalar SGS preconditioner is not the same as PETSc's.
  #add_test(NAME MPIPetsc-CSR-SGS 
  # COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type aij
  #	)
  ## Fails! Our scalar SGS relaxation is not the same as PETSc's.
  #add_test(NAME MPIPetsc-CSR-SGS-Relaxation 
  # COMMAND env OMP_NUM_THREADS=1 ${MPIEXEC} -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
  #	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
  #	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/block4_relaxation_compare.perc -mat_type aij
  #	-ref_sub_pc_type sor -ref_sub_pc_sor_local_symmetric -blasted_pc_type sgs
  #	)

  add_test(NAME AsyncRelaxation-Petsc-BSR4-GS-Convergence 
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testthreadedpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type gs -test_type convergence
	)
  add_test(NAME AsyncRelaxation-Petsc-BSR4-SGS-IterationsLowerBound 
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testthreadedpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncrelaxation.perc -mat_type baij
	-ref_sub_pc_type pbjacobi -blasted_pc_type sgs -test_type upper_bound_its
	)

  add_test(NAME AsyncPreconditioning-Petsc-BSR4-SGS
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testthreadedpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncpreconditioning.perc -mat_type baij
	-ref_sub_pc_type sor -blasted_pc_type sgs -blasted_async_sweeps 1,4
	-test_type compare_its
	)

  add_test(NAME AsyncPreconditioning-Petsc-BSR4-ILU0
	COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testthreadedpetscsolver
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/asyncpreconditioning.perc -mat_type baij
	-ref_sub_pc_type ilu -blasted_pc_type ilu0 -blasted_async_sweeps 2,3
	-test_type compare_its
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
