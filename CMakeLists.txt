cmake_minimum_required(VERSION 3.27)
project(rall)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)
add_executable(test test/test.cc)