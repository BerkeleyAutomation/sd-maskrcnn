find_package( JPEG REQUIRED )
find_package( PNG REQUIRED )

include_directories( ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIR} )

add_library( imgproc color.cpp filter.cpp gradient.cpp morph.cpp nms.cpp resample.cpp image.cpp )
target_link_libraries( imgproc util ${JPEG_LIBRARIES} ${PNG_LIBRARIES} )
