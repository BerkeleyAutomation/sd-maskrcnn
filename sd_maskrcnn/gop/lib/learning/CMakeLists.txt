add_library( learning splitcriterion.cpp treedata.cpp tree.cpp forest.cpp features.cpp )
target_link_libraries( learning util )
