cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME Lab_5)
project(${PROJECT_NAME} C)

set(CMAKE_C_STANDARD 11)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

set(SRC_LIST list.c)
set(HEADER_LIST list.h)

add_library(list SHARED ${SRC_LIST})

add_executable(${PROJECT_NAME} main.c)

# Линкуем библиотеку к исполняемому файлу
target_link_libraries(${PROJECT_NAME} PRIVATE list)

# Добавляем пути для заголовочных файлов
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Установка переменной среды для поиска библиотек во время выполнения
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Настройка переменной среды для поиска библиотеки
set_target_properties(list PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

