add_executable(testhelper testhelper.cpp)
target_link_libraries(testhelper helper)

add_test(NAME Unit_SortBlockInner
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testhelper
  ${CMAKE_CURRENT_SOURCE_DIR}/input/unsorted_arrays.txt
  ${CMAKE_CURRENT_SOURCE_DIR}/input/sorted_arrays.txt
  )
