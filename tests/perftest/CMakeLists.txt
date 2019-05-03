set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(perftest runperftest.cpp perftesting.cpp)
target_link_libraries(perftest blasted_petsc)

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
