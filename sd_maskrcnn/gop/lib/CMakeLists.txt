include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(imgproc)
add_subdirectory(contour)
add_subdirectory(learning)
add_subdirectory(proposals)
add_subdirectory(segmentation)
add_subdirectory(util)

if( USE_PYTHON )
	add_subdirectory(python)
endif()