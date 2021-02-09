add_executable(test_prefix_sum test_prefix_sum.cpp)
target_link_libraries(test_prefix_sum blasted_utils)

add_test(NAME Unit_PrefixSum
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_prefix_sum
  )
