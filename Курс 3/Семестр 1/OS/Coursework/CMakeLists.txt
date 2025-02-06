cmake_minimum_required(VERSION 3.10)

project(OS_Coursework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(PcapPlusPlus REQUIRED)

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")

add_executable("${PROJECT_NAME}" ${SOURCES})

set_target_properties(
  "${PROJECT_NAME}"
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
             OUTPUT_NAME "OS_Coursework.out")

target_include_directories("${PROJECT_NAME}" PUBLIC ${INCLUDE_DIR})

target_link_libraries(
  "${PROJECT_NAME}" PUBLIC PcapPlusPlus::Pcap++ PcapPlusPlus::Packet++
                           PcapPlusPlus::Common++)
