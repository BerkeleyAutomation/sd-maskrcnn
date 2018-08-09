# Setup MEX to compile our code
SET( CMAKE_CXX_COMPILER mex )
SET( CMAKE_C_COMPILER   mex )

# Suffix and Prefix of the output target file
# TODO: Make this platform independent
SET( CMAKE_SHARED_LIBRARY_SUFFIX .mexa64 )
SET( CMAKE_SHARED_LIBRARY_PREFIX )

# Setup the Build flags
SET( CMAKE_C_FLAGS "" )
SET( CMAKE_CXX_FLAGS "-cxx -largeArrayDims CXXFLAGS='$$CXXFLAGS -std=c++11 -D_GLIBCXX_USE_NANOSLEEP'" )
SET( CMAKE_CXX_FLAGS_RELEASE "" )          # Remove any -O flags
SET( CMAKE_POSITION_INDEPENDENT_CODE OFF ) # Remove the -fPIC flag
SET( CMAKE_SHARED_LIBRARY_C_FLAGS "" )     # Remove the -fPIC flag
SET( CMAKE_SHARED_LIBRARY_CXX_FLAGS "" )   # Remove the -fPIC flag
SET( CMAKE_SHARED_MODULE_C_FLAGS "" )      # Remove the -fPIC flag
SET( CMAKE_SHARED_MODULE_CXX_FLAGS "" )    # Remove the -fPIC flag

SET( CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cpp ).o <OBJECT>" )

# Variables controlling the linking-phase
SET( CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS )    # remove -shared options. mex does not accept the "-shared" option
SET( CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>" )

# Variables controlling the installation RPATH
SET( CMAKE_INSTALL_RPATH "\$ORIGIN" )
# CMake will reset RPATH at the installation phase, so we need to specify CMAKE_INSTALL_RPATH
MESSAGE( "${CMAKE_CXX_FLAGS_RELEASE}" )

include_directories( ../lib )
add_library( gop_mex SHARED gop_mex.cpp )
target_link_libraries( gop_mex util imgproc learning contour segmentation proposals gomp )
