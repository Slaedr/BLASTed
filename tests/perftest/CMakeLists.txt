set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(perftest_petsc runperftest_petsc.cpp perftesting_petsc.cpp)
target_link_libraries(perftest_petsc blasted_petsc)

add_executable(perftest runperftest.cpp perftesting.cpp)
target_link_libraries(perftest blockmatrices solverops ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
