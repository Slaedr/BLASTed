# test executables

add_library(solvers solvers.cpp)

add_executable(testbsrmatrix runbsrmatrixtest.cpp testbsrmatrix.cpp)
target_link_libraries(testbsrmatrix coomatrix blockmatrices)

add_executable(testcsrmatrix runcsrmatrixtest.cpp testcsrmatrix.cpp)
target_link_libraries(testcsrmatrix coomatrix blockmatrices)

add_executable(testcoomatrix runcoomatrixtest.cpp testcoomatrix.cpp testcsrmatrix.cpp)
target_link_libraries(testcoomatrix coomatrix blockmatrices)

add_executable(testsolve runsolvetest.cpp testsolve.cpp)
target_link_libraries(testsolve solvers coomatrix blockmatrices)

# Tests

# valgrind --leak-check=full 

add_test(NAME COORead COMMAND testcoomatrix read
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted.coo)

add_test(NAME COOConvertToCSR COMMAND testcoomatrix convertCSR 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix_sorted.coo)

add_test(NAME COOConvertToBSR3 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcoomatrix convertBSR3 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted_bcolmajor.bcoo)

add_test(NAME CSRMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply matrix 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)
add_test(NAME CSRViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply view
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)

add_test(NAME BSR3ViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 3
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_b.mtx
)

add_test(NAME BSR7MatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply matrix rowmajor 7
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)
add_test(NAME BSR7ViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view rowmajor 7
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)
add_test(NAME BSR7ColViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 7
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)

add_test(NAME SPDCSRJacobi COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
	1e-10 1e-9 200
)

add_test(NAME SPDCSRSGS COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
	1e-10 1e-10 200
)

add_test(NAME SPDCSRILU0 COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/boeing-msc00726/msc00726_b.mtx
	1e-10 1e-10 200
)

add_test(NAME CSRJacobi COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)
add_test(NAME CSRSGS COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)
add_test(NAME CSRILU0 COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 csr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4JacobiRowmajor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi bsr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4SGSRowmajor COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs bsr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4ILU0Rowmajor COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 bsr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4NoneColmajor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs none bsr colmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-11 1e-8 1000
)

add_test(NAME BSR4JacobiColmajor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs jacobi bsr colmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4SGSColmajor COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs sgs bsr colmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

add_test(NAME BSR4ILU0Colmajor COMMAND env OMP_NUM_THREADS=1 ${CMAKE_CURRENT_BINARY_DIR}/testsolve bcgs ilu0 bsr colmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.mtx
	1e-10 1e-8 200
)

if(WITH_PETSC)
	add_executable(testpetscsolver runpetsctest.c)
	target_link_libraries(testpetscsolver coomatrix blasted_petsc)
	
	add_test(NAME SerialPetsc-BSR4-Jacobi COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_jacobi.perc
	)
	add_test(NAME SerialPetsc-BSR4-SGS COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type baij
	)
	add_test(NAME SerialPetsc-BSR4-ILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type baij
	)
	add_test(NAME SerialPetsc-CSR-ILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 1 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type aij
	)
	add_test(NAME MPIPetsc-BSR4-Jacobi COMMAND env OMP_NUM_THREADS=1 mpirun -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_jacobi.perc
	)
	add_test(NAME MPIPetsc-BSR4-SGS COMMAND env OMP_NUM_THREADS=1 mpirun -n 2 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs.perc -mat_type baij
	)
	add_test(NAME MPIPetsc-BSR4-ILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
		${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
		-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0.perc -mat_type baij
	)
	
	#add_test(NAME MPIPetscMKL-BSR4-SGS COMMAND env OMP_NUM_THREADS=1 mpirun -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	#	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_sgs_mkl.perc
	#)
	#add_test(NAME MPIPetscMKL-BSR4-ILU0 COMMAND env OMP_NUM_THREADS=1 mpirun -n 3 ${CMAKE_CURRENT_BINARY_DIR}/testpetscsolver
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1.pmat
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_b.pmat 
	#	${CMAKE_CURRENT_SOURCE_DIR}/input/fvens-2dcyl1/2dcyl1_x.pmat
	#	-options_file ${CMAKE_CURRENT_SOURCE_DIR}/input/2dcyl1_ilu0_mkl.perc
	#)
	
endif()

add_subdirectory(poisson3d-fd)
