# test executables

add_library(testlibcsrmatrix testcsrmatrix.cpp)
add_library(testlibbsrmatrix testbsrmatrix.cpp)
add_library(testlibsolve testsolve.cpp)

add_executable(testbsrmatrix bsrmatrix.cpp)
target_link_libraries(testbsrmatrix testlibbsrmatrix)
add_executable(testcsrmatrix csrmatrix.cpp)
target_link_libraries(testcsrmatrix testlibcsrmatrix)
add_executable(testcoomatrix coomatrix.cpp testcoomatrix.cpp)
target_link_libraries(testcoomatrix testlibcsrmatrix)
add_executable(testsolve solve.cpp)
target_link_libraries(testsolve testlibsolve)

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

add_subdirectory(poisson3d-fd)
