add_library( segmentation aggregation.cpp iouset.cpp segmentation.cpp )
target_link_libraries( segmentation util learning imgproc contour )
