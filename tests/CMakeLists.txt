# test executables

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(testbsrmatrix bsrmatrix.cpp)
add_executable(testcoomatrix coomatrix.cpp)

#add_executable(testbsr_petsc bsr_petsc.cpp)
#target_link_libraries(testbsr_petsc bsr_petsc)

# Tests

enable_testing()

#add_test(NAME BSRMatMul COMMAND testbsrmatrix ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat matmul)
add_test(NAME COOMatrixMarketRead COMMAND testcoomatrix ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx read)
