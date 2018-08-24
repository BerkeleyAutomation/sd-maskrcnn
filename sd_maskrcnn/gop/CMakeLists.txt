cmake_minimum_required(VERSION 2.8)
project( pseg )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Making MAC OS shut up
if(APPLE AND NOT DEFINED CMAKE_MACOSX_RPATH)
	set(CMAKE_MACOSX_RPATH 0)
endif()

# Checking for OpenMP
find_package( OpenMP )
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MSVC)
	set(Boost_USE_STATIC_LIBS       ON)
	set(Boost_USE_MULTITHREADED     OFF)
	set(Boost_USE_STATIC_RUNTIME    OFF)
	add_definitions(-DBOOST_ALL_NO_LIB)
	add_definitions(-DBOOST_PYTHON_STATIC_LIB)
endif()

# Finding python and boost-python
option( USE_PYTHON "Enable the python wrapper" OFF )

# Finding Eigen
find_package( Eigen3 3.2.0 )
if( NOT EIGEN3_FOUND )
	message( WARNING "No system eigen library found, trying the external/eigen directory" )
	set( EIGEN3_INCLUDE_DIR "external/eigen" )
	
	file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen3_version_header)
	
	string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen3_world_version_match "${_eigen3_version_header}")
	set(EIGEN3_WORLD_VERSION "${CMAKE_MATCH_1}")
	string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen3_major_version_match "${_eigen3_version_header}")
	set(EIGEN3_MAJOR_VERSION "${CMAKE_MATCH_1}")
	string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen3_minor_version_match "${_eigen3_version_header}")
	set(EIGEN3_MINOR_VERSION "${CMAKE_MATCH_1}")

	set(EIGEN3_VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})
	if(${EIGEN3_VERSION} VERSION_LESS "3.2.0")
		message( FATAL_ERROR "Eigen 3.2 or newer required, you can either install it on your system or copy it into the external/eigen directory" )
	endif()
	set( EIGEN3_FOUND On )
endif()

include_directories( ${EIGEN3_INCLUDE_DIR} )

if( CMAKE_VERSION VERSION_LESS "2.8.9" )
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
else()
	set(CMAKE_POSITION_INDEPENDENT_CODE ON) 
endif()
# Setup some unix build flags and enable c++11
if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-sign-compare -pedantic") 
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DEIGEN_DONT_PARALLELIZE")
include_directories( external/ external/CImg/ external/rapidxml-1.13/ external/libsvm-3.17/ external/liblbfgs-1.10/include external/maxflow-v3.03.src/ external/ibfs/ external/quadprog/ )
add_definitions(-DLBFGS_FLOAT=32)


# Build MATLAB
option( USE_MATLAB "Enable the matlab wrapper" OFF )
if(USE_MATLAB)
	add_subdirectory( matlab )
endif()


add_subdirectory( lib )
add_subdirectory( external )
add_subdirectory( examples )

