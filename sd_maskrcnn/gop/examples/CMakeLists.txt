include_directories( ../lib )
add_executable( example example.cpp )
target_link_libraries( example imgproc proposals contour )

add_executable( example_learned example_learned.cpp )
target_link_libraries( example_learned imgproc proposals contour )
