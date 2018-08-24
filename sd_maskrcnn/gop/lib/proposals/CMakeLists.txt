add_library( proposals saliency.cpp objectives.cpp unaryfeature.cpp unary.cpp edgefeature.cpp proposal.cpp geodesics.cpp seedfeature.cpp seed.cpp edgeweight.cpp )
target_link_libraries( proposals util learning imgproc contour lbfgs segmentation )
