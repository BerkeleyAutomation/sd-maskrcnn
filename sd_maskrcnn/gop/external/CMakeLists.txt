if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

if( USE_PYTHON )
	add_subdirectory( Boost.NumPy )
endif()

add_subdirectory( libsvm-3.17 )
add_subdirectory( liblbfgs-1.10 )