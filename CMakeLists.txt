cmake_minimum_required(VERSION 3.5)

project(align2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(WITH_TESTS "Enable tests" OFF)

#set (Python3_FIND_ABI "ON" "ANY" "ANY")
find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED core imgproc)
if(WITH_TESTS)
    find_package(GTest)
endif()


add_library(${PROJECT_NAME} SHARED
    src/main.cpp src/jacobi.cpp src/aligner.cpp
    include/jacobi.h include/utils.h include/pyramid.h include/aligner.h)
target_include_directories(align2d PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Python3::Module Python3::NumPy Eigen3::Eigen ${OpenCV_LIBS})

set(SOABI "cpython-38-x86_64-linux-gnu")
set_property (TARGET ${PROJECT_NAME} PROPERTY OUTPUT_NAME "${PROJECT_NAME}.${SOABI}")
set_property (TARGET ${PROJECT_NAME} PROPERTY PREFIX "")

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    configure_file(test.py test.py COPYONLY)
ENDIF()
