file(GLOB_RECURSE LINEARALGEBRA_LIB_SRC *.cpp 
                                        *.cu )

add_library(LinearAlgebra STATIC ${LINEARALGEBRA_LIB_SRC})

target_include_directories(LinearAlgebra PUBLIC .)