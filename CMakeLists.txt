cmake_minimum_required(VERSION 3.16)
project(jetson_llm_harness CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(llm_experiments src/main.cpp)
target_include_directories(llm_experiments PRIVATE src)
