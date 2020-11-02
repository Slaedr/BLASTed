add_library(blasted_utils prefix_sum.cpp)

if(WITH_PETSC)
  add_library(utils petsc_cmdoptions.cpp)
  target_link_libraries(utils ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

  add_executable(util_petsc_io util_petsc_io.c blasted_petsc_io.cpp)
  target_link_libraries(util_petsc_io coomatrix blockmatrices
  	${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

  add_executable(convert_to_mtx convert_to_mtx.cpp)
  target_link_libraries(convert_to_mtx ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
endif()
