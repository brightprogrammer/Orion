cmake_minimum_required(VERSION 3.5)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# project settings
project(ORION LANGUAGES CXX)
set(CMAKE_CXX_VERSION 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic \
-Wconversion -Wcast-align -Wunused -Wshadow -Wold-style-cast \
-Wpointer-arith -Wcast-qual -Wno-missing-braces")

add_executable(test test.cpp)
