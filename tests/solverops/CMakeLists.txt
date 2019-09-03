
add_executable(test_iterprec_conv test_iter_prec_conv.cpp async_triangular_factors_convergence.cpp
  async_ilu_convergence.cpp)
target_link_libraries(test_iterprec_conv poisson solverops rawmatrixutils)

add_test(NAME AsyncILU-SweepsConvergence-CSR-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type aij -blasted_thread_chunk_size 128 -max_sweeps 150 -tolerance 1e-15
  -initialization orig -blasted_use_symmetric_scaling false -num_repeats 2
  )

add_test(NAME AsyncILU-Scaled-SweepsConvergence-CSR-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type aij -blasted_thread_chunk_size 128 -max_sweeps 150 -tolerance 1e-15
  -initialization orig -blasted_use_symmetric_scaling true -num_repeats 2
  )

add_test(NAME AsyncILU-SweepsConvergence-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 32 -max_sweeps 150 -tolerance 1e-15
  -initialization orig -blasted_use_symmetric_scaling false -num_repeats 2
  )

add_test(NAME AsyncILU-Scaled-SweepsConvergence-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 32 -max_sweeps 150 -tolerance 1e-15
  -initialization orig -blasted_use_symmetric_scaling true -num_repeats 2
  )

add_test(NAME AsyncILU-ExactFixedPoint-CSR-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type aij -blasted_thread_chunk_size 64 -max_sweeps 5 -tolerance 1e-16
  -initialization exact -blasted_use_symmetric_scaling false -num_repeats 2
  )

add_test(NAME AsyncILU-ExactFixedPoint-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 16 -max_sweeps 5 -tolerance 1e-16
  -initialization exact -blasted_use_symmetric_scaling false -num_repeats 2
  )

add_test(NAME AsyncILU-Scaled-ExactFixedPoint-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv ailu file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 16 -max_sweeps 5 -tolerance 1e-16
  -initialization exact -blasted_use_symmetric_scaling true -num_repeats 2
  )

add_test(NAME AsyncILUTriangular-ExactFixedPoint-CSR-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv triangular file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type aij -blasted_thread_chunk_size 64 -max_sweeps 5 -tolerance 1e-16
  -initialization exact -blasted_use_symmetric_scaling false -num_repeats 1
  )

add_test(NAME AsyncILUTriangular-ExactFixedPoint-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv triangular file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 16 -max_sweeps 5 -tolerance 1e-16
  -initialization exact -blasted_use_symmetric_scaling false -num_repeats 1
  )

add_test(NAME AsyncILUTriangular-SweepsConvergence-CSR-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv triangular file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type aij -blasted_thread_chunk_size 128 -max_sweeps 150 -tolerance 1e-15
  -initialization zero -blasted_use_symmetric_scaling false -num_repeats 1
  )

add_test(NAME AsyncILUTriangular-SweepsConvergence-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv triangular file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 32 -max_sweeps 150 -tolerance 1e-15
  -initialization zero -blasted_use_symmetric_scaling false -num_repeats 1
  )
add_test(NAME AsyncILUTriangular-Scaled-SweepsConvergence-BSR4-2dcyl
  COMMAND ${SEQEXEC} ${THREADOPTS} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/test_iterprec_conv triangular file
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_b.pmat
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1_x.pmat
  -mat_type baij -blasted_thread_chunk_size 32 -max_sweeps 150 -tolerance 1e-15
  -initialization zero -blasted_use_symmetric_scaling true -num_repeats 1
  )
