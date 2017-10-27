# test executables

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/tests)

add_executable(bsrmatmul bsrmatmul.cpp)

#add_executable(bsr_cinterface cbsr.cpp)
#target_link_libraries(bsrmatmul fvens_base)

# Tests

enable_testing()

add_test(NAME BSRMatMul COMMAND bsrmatmul ${CMAKE_CURRENT_SOURCE_DIR}/input/matrix.dat)
