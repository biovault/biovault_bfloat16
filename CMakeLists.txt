# Niels Dekker, LKEB, Leiden University Medical Center, 2020

# At the moment of writing (July 2020), the Microsoft-hosted VM ubuntu-18.04
# still runs CMake 3.17.0 on Azure Pipelines.
cmake_minimum_required( VERSION 3.17.0 )

project(biovault_bfloat16)

message(STATUS "[${PROJECT_NAME}] CMAKE_VERSION = ${CMAKE_VERSION}")
message(STATUS "[${PROJECT_NAME}] CMAKE_GENERATOR = ${CMAKE_GENERATOR}")

# Use C++14, to allow binary literals (among other C++14 features) to be used in the unit tests.
set(CMAKE_CXX_STANDARD 14)

# No /nologo for Visual C++
set(CMAKE_VERBOSE_MAKEFILE ON)

# Download and build GoogleTest as described at
# https://github.com/google/googletest/blob/release-1.10.0/googletest/README.md

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

add_executable(${PROJECT_NAME}_test
  biovault_bfloat16.h
  biovault_bfloat16_test.cpp
)
target_link_libraries(${PROJECT_NAME}_test gtest_main)

# From https://stackoverflow.com/questions/2368811/how-to-set-warning-level-in-cmake/50882216#50882216
# by mrts, 15 June 2018
if(MSVC)
  target_compile_options(${PROJECT_NAME}_test PRIVATE /W4 /WX)
else()
  target_compile_options(${PROJECT_NAME}_test PRIVATE -Wall -Wextra -pedantic -Werror -Wfloat-equal)
endif()

add_test(NAME bfloat16_test COMMAND ${PROJECT_NAME})
