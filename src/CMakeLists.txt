if(WITH_PETSC)
	find_package(MPI REQUIRED)

	add_library(blasted_petsc blasted_petsc.cpp)
	target_link_libraries(blasted_petsc ${PETSC_LIB} ${MPI_CXX})
	target_include_directories(blasted_petsc PRIVATE ${MPI_C_INCLUDE_PATH})
endif()

# Tags

set_source_files_properties(tags PROPERTIES GENERATED true)
add_custom_target ( tags
    COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
  )

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
    message(STATUS "Found ctags: ${CTAGS_PATH}")
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)


