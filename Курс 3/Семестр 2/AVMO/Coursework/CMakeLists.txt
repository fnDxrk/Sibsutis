cmake_minimum_required(VERSION 3.10)
project(AVMO_Coursework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)