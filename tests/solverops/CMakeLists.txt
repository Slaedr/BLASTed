
add_executable(test_ailu_conv async_ilu_convergence.cpp)
target_link_libraries(test_ailu_conv poisson solverops rawmatrixutils)

add_test(NAME AsyncILU-SweepsConvergence-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_ailu_conv file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat 
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 128 -max_sweeps 500 -tolerance 1e-8
  )
