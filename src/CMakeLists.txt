
add_library(cblasted cblasted.cpp)

if(WITH_PETSC)
	target_link_libraries(cblasted ${PETSC_LIB})
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
    add_dependencies (cblasted tags)
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)


