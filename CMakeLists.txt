cmake_minimum_required (VERSION 2.6)
project (my1stpca)

add_executable(my1stpca src/main.cpp )

target_include_directories(my1stpca PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

