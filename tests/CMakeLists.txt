# test executables

add_library(solvers solvers.cpp)

add_executable(testbsrmatrix runbsrmatrixtest.cpp testbsrmatrix.cpp)
add_executable(testcsrmatrix runcsrmatrixtest.cpp testcsrmatrix.cpp)
add_executable(testcoomatrix runcoomatrixtest.cpp testcoomatrix.cpp testcsrmatrix.cpp)
add_executable(testsolve runsolvetest.cpp testsolve.cpp)
target_link_libraries(testsolve solvers)

# Tests

add_test(NAME COORead COMMAND testcoomatrix read 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted.coo)

add_test(NAME COOConvertToCSR COMMAND testcoomatrix convertCSR ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix_sorted.coo)

add_test(NAME COOConvertToBSR3 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcoomatrix convertBSR3 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted_bcolmajor.bcoo)

add_test(NAME CSRMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply matrix 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)
add_test(NAME CSRViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply view
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)

#add_test(NAME BSR3MatMul COMMAND testbsrmatrix apply matrix rowmajor 
#	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_x.mtx 
#	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_b.mtx
#)
#add_test(NAME BSR3ViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view rowmajor
#	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
#	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_x.mtx 
#	${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_b.mtx
#)

# valgrind --leak-check=full 

add_test(NAME BSR7MatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply matrix rowmajor 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)
add_test(NAME BSR7ViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)
add_test(NAME BSR7ColViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)

add_test(NAME BSR7ViewILU0Rowmajor COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testsolve ilu0 bsr rowmajor
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_x.mtx 
	${CMAKE_CURRENT_SOURCE_DIR}/input/cfd-matrix-test/DK01R_b.mtx
)

add_subdirectory(poisson3d-fd)
