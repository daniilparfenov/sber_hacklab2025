cmake_minimum_required(VERSION 3.10)
project(hackaton2025)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags equivalent to your Makefile
add_compile_options(-O3 -DNDEBUG -Wall -Wextra -fopenmp)
add_link_options(-flto -O3 -fopenmp)

# Source discovery (equivalent to wildcard)
file(GLOB HACK_SRCS hackathon_internals/*.cpp)
file(GLOB SRC_SRCS src/*.cpp)

# Entry point (interface.cpp)
set(ENTRY_POINT interface.cpp)

# Create executable
add_executable(app
        ${HACK_SRCS}
        ${SRC_SRCS}
        ${ENTRY_POINT}
)

# Include directories
target_include_directories(app PRIVATE
        hackathon_internals
        src
)
