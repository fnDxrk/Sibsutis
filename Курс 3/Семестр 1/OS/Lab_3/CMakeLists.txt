cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME Lab_3)
project(${PROJECT_NAME} C)

set(CMAKE_C_STANDARD 11)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

add_executable(${PROJECT_NAME} main.c)

