add_library(coomatrix coomatrix.cpp)
add_library(blockmatrices csrmatrix.cpp bsrmatrix.cpp)
set_property(TARGET blockmatrices PROPERTY POSITION_INDEPENDENT_CODE ON)

if(WITH_PETSC)
	#find_package(MPI REQUIRED)

	add_library(blasted_petsc SHARED blasted_petsc.cpp)
	target_link_libraries(blasted_petsc ${PETSC_LIB} blockmatrices)
	
	add_executable(util_petsc_io util_petsc_io.c blasted_petsc_io.cpp)
	target_link_libraries(util_petsc_io coomatrix blockmatrices ${PETSC_LIB})
	
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


