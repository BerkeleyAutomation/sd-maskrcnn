# - Find nana
# Find the native ANN headers and libraries.
#
#  MATIO_INCLUDE_DIRS - where to find nana/nana.h, etc.
#  MATIO_LIBRARIES    - List of libraries when using nana.
#  MATIO_FOUND        - True if nana found.

# Look for the header file.
FIND_PATH(MATIO_INCLUDE_DIR NAMES matio.h)
MARK_AS_ADVANCED(MATIO_INCLUDE_DIR)

# Look for the library.
FIND_LIBRARY(MATIO_LIBRARY NAMES matio)
MARK_AS_ADVANCED(MATIO_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set MATIO_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MATIO DEFAULT_MSG MATIO_LIBRARY MATIO_INCLUDE_DIR)

IF(MATIO_FOUND)
  SET(MATIO_LIBRARIES ${MATIO_LIBRARY})
  SET(MATIO_INCLUDE_DIRS ${MATIO_INCLUDE_DIR})
ELSE(MATIO_FOUND)
  SET(MATIO_LIBRARIES)
  SET(MATIO_INCLUDE_DIRS)
ENDIF(MATIO_FOUND)
