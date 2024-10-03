cmake_minimum_required(VERSION 3.10)

project(OS_Coursework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.h")

add_executable(OS_Coursework ${SOURCES})
