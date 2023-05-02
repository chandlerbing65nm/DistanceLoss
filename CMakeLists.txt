cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(distance_loss LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

find_package(Python3 COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 development headers not found")
endif()

set(SOURCE_FILES
    src/distance_loss.cpp
    src/distance_loss.cu
)

add_library(distance_loss SHARED ${SOURCE_FILES})

target_include_directories(distance_loss PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_libraries(distance_loss PRIVATE ${TORCH_LIBRARIES})
target_compile_options(distance_loss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>)

install(TARGETS distance_loss DESTINATION ${CMAKE_SOURCE_DIR}/python)
