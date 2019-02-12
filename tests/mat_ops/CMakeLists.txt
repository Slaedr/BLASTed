# test executables

add_executable(testbsrmatrix runbsrmatrixtest.cpp testbsrmatrix.cpp)
target_link_libraries(testbsrmatrix coomatrix blockmatrices)

add_executable(testcsrmatrix runcsrmatrixtest.cpp testcsrmatrix.cpp)
target_link_libraries(testcsrmatrix coomatrix blockmatrices)

add_executable(testcoomatrix runcoomatrixtest.cpp testcoomatrix.cpp testcsrmatrix.cpp)
target_link_libraries(testcoomatrix coomatrix blockmatrices)

add_executable(testcscmatrix scmatrixtest.cpp)
target_link_libraries(testcscmatrix blockmatrices coomatrix)

add_executable(testbscconversion bscconversiontest.cpp)
target_link_libraries(testbscconversion blockmatrices coomatrix rawmatrixutils)

add_executable(testlevelschedule testlevelschedule.cpp)
target_link_libraries(testlevelschedule coomatrix solverops)

add_executable(testreordering testreordering.cpp)
target_link_libraries(testreordering orderingscaling blockmatrices rawmatrixutils coomatrix helper)
if(WITH_MC64)
  target_link_libraries(testreordering ${MC64_LIB} ${FORTLDFLAGS})
endif(WITH_MC64)

if(WITH_MC64)
  add_executable(testmc64 testmc64.cpp)
  target_link_libraries(testmc64 orderingscaling blockmatrices coomatrix rawmatrixutils helper)
  target_link_libraries(testmc64 ${MC64_LIB} ${FORTLDFLAGS})
endif(WITH_MC64)

# Tests

add_test(NAME COORead COMMAND ${SEQEXEC} ${SEQTASKS} testcoomatrix read
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted.coo)

add_test(NAME COOConvertToCSR
  COMMAND ${SEQEXEC} ${SEQTASKS} testcoomatrix convertCSR
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_coo_matrix_sorted.coo)

add_test(NAME COOConvertToBSR3
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcoomatrix convertBSR3
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_sorted_bcolmajor.bcoo)

add_test(NAME CSRMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply matrix
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME CSRViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testcsrmatrix apply view
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )

add_test(NAME BSR3ViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 3
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix_b.mtx
  )

add_test(NAME BSR7MatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply matrix rowmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME BSR7ViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view rowmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
  )
add_test(NAME BSR7ColViewMatMul
  COMMAND ${SEQEXEC} ${SEQTASKS} ${CMAKE_CURRENT_BINARY_DIR}/testbsrmatrix apply view colmajor 7
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx
)

add_test(NAME ConvertCSRToCSC COMMAND ${SEQEXEC} ${SEQTASKS} testcscmatrix 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.csc)

add_test(NAME ConvertBSRToBSC_3 COMMAND ${SEQEXEC} ${SEQTASKS} testbscconversion
  ${CMAKE_CURRENT_SOURCE_DIR}/input/small_block3_matrix.mtx 3)

add_test(NAME ConvertBSRToBSC_7 COMMAND ${SEQEXEC} ${SEQTASKS} testbscconversion
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 7)

add_test(NAME B7VecApplyRowReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx 7 row vector
  )
add_test(NAME B7VecApplyColReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_b.mtx 7 column vector
  )
add_test(NAME B1VecApplyRowReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 1 row vector
  )
add_test(NAME B1VecApplyColReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R_x.mtx 1 column vector
  )

add_test(NAME BSR7ApplyRowReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 7 row matrix
  )
add_test(NAME BSR7ApplyColReordering COMMAND ${SEQEXEC} ${SEQTASKS} testreordering
  ${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 7 column matrix
  )

add_test(NAME LevelSchedule_Blk4 COMMAND ${SEQEXEC} ${SEQTASKS} testlevelschedule
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.mtx 4
  )

add_test(NAME LevelSchedule_1 COMMAND ${SEQEXEC} ${SEQTASKS} testlevelschedule
  ${CMAKE_CURRENT_SOURCE_DIR}/../input/boeing-msc00726/msc00726.mtx 1
  )

if(WITH_MC64)
  add_test(NAME MC64Job_1_DK01R COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 1
	)
  add_test(NAME MC64Job_1_msc00726 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 1
	)
  # add_test(NAME MC64Job_1_2dcyl1 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
  # 	${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.mtx 1
  # 	)
  add_test(NAME MC64Job_2_DK01R COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 2
	)
  add_test(NAME MC64Job_2_msc00726 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/../input/boeing-msc00726/msc00726.mtx 2
	)
  # add_test(NAME MC64Job_2_2dcyl1 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
  # 	${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.mtx 2
  # 	)
  add_test(NAME MC64Job_3_DK01R COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 3
	)
  add_test(NAME MC64Job_3_mcs00726 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/../input/boeing-msc00726/msc00726.mtx 3
	)
  # add_test(NAME MC64Job_3_2dcyl1 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
  # 	${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.mtx 3
  # 	)
  add_test(NAME MC64Job_4_DK01R COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 4
	)
  add_test(NAME MC64Job_4_msc00726 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
	${CMAKE_CURRENT_SOURCE_DIR}/../input/boeing-msc00726/msc00726.mtx 4
	)
  # add_test(NAME MC64Job_4_2dcyl1 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
  # 	${CMAKE_CURRENT_SOURCE_DIR}/../input/fvens-2dcyl1/2dcyl1.mtx 4
  # 	)
  # add_test(NAME MC64Job_5 COMMAND ${SEQEXEC} ${SEQTASKS} testmc64
  # 	${CMAKE_CURRENT_SOURCE_DIR}/input/fluorem-dk01r/DK01R.mtx 5
  # 	)
endif(WITH_MC64)
