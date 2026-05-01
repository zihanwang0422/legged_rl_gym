cmake_minimum_required(VERSION 3.10)
project(unitree_legged_sdk_python_wrapper)

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
    set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
    set(ARCH arm64)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib/cpp/${ARCH})
file(GLOB_RECURSE sources "cpp/*.cpp")

set(EXTRA_LIBS -pthread rt m libunitree_legged_sdk.a)
# set(CMAKE_CXX_FLAGS "-O3 -no-pie")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fPIC")
set(CMAKE_CXX_STANDARD 14)
# python
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib/python/${ARCH}) # .so .dylib

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "Found system pybind11: using it")
else()
    message(STATUS "System pybind11 not found: using bundled third-party/pybind11")
    add_subdirectory(third-party/pybind11)
endif()

pybind11_add_module(robot_interface python_interface.cpp)
target_link_libraries(robot_interface PRIVATE ${EXTRA_LIBS} unitree_legged_sdk)

