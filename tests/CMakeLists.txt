# test executables

add_library(testlibcsrmatrix testcsrmatrix.cpp)
add_library(testlibbsrmatrix testbsrmatrix.cpp)

add_executable(testbsrmatrix bsrmatrix.cpp)
target_link_libraries(testbsrmatrix testlibbsrmatrix)
add_executable(testcsrmatrix csrmatrix.cpp)
target_link_libraries(testcsrmatrix testlibcsrmatrix)
add_executable(testcoomatrix coomatrix.cpp testcoomatrix.cpp)
target_link_libraries(testcoomatrix testlibcsrmatrix)

# Tests

add_test(NAME COORead COMMAND testcoomatrix read 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted.coo)

add_test(NAME COOConvertToCSR COMMAND testcoomatrix convertCSR ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix_sorted.coo)

add_test(NAME COOConvertToBSR3 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testcoomatrix convertBSR3 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted_bcolmajor.bcoo)

add_test(NAME CSRMatMul COMMAND testcsrmatrix apply matrix ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat)
add_test(NAME CSRViewMatMul COMMAND testcsrmatrix apply view ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat)

add_test(NAME BSRMatMul COMMAND testbsrmatrix apply matrix ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat)
add_test(NAME BSRViewMatMul COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix 
	apply view ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat)

add_subdirectory(poisson3d-fd)
