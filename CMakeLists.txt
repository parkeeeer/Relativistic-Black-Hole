cmake_minimum_required(VERSION 4.0)
project(RelatavisticGravity)

set(CMAKE_CXX_STANDARD 23)


find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(RelatavisticGravity main.cpp)


target_link_libraries(RelatavisticGravity PRIVATE OpenGL::GL glfw glm::glm)

target_compile_definitions(RelatavisticGravity PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")