cmake_minimum_required(VERSION 3.25)

project(neural)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXAMPLES_PATH examples)
set(SRC_PATH src)
set(INCLUDE_PATH include)
set(THIRD_PARTY third_party)
set(TEST_PATH test)
set(CMAKE_C_STANDARD 17)

include_directories(${INCLUDE_PATH})
include_directories(${THIRD_PARTY})
file(GLOB source_files ${SRC_PATH}/*.c)
add_library(neural ${source_files})

file(GLOB examples ${EXAMPLES_PATH}/*)
foreach(example ${examples})
  get_filename_component(example_name ${example} NAME_WE)
  message("Building: ${example}")
  file(GLOB sources ${example}/*.c)
  add_executable(${example_name} ${sources})
  target_link_libraries(${example_name} neural -lm)
endforeach()


file(GLOB test_files ${TEST_PATH}/*.c)

add_executable(run_tests ${test_files})
target_link_libraries(run_tests neural -lm)
