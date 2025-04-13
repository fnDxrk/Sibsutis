cmake_minimum_required(VERSION 3.28)
project(lab10 CUDA)


set(TARGET1_SRC "vector_product/vector_product.cu")
set(TARGET2_SRC "matrix_transpose/matrix_transpose.cu")

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(scalar_vector_product ${TARGET1_SRC})
add_executable(matrix_transposition ${TARGET2_SRC})

set_target_properties(scalar_vector_product PROPERTIES CUDA_ARCHITECTURES "89")
set_target_properties(matrix_transposition PROPERTIES CUDA_ARCHITECTURES "89")

target_link_libraries(scalar_vector_product ${CUDA_LIBRARIES})
target_link_libraries(matrix_transposition  ${CUDA_LIBRARIES})