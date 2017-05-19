cmake_minimum_required(VERSION 3.5.1)
project(sphere_sampling)

#c++ 11
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(glfw3 3.2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Assimp REQUIRED)

add_library(ss ss.cpp)

#add_executable(sphere_sampling main.cpp)

##target_include_directories(sph ${OPENGL_INCLUDE_DIR})
#target_link_libraries(sphere_sampling ${OPENGL_gl_LIBRARY})
#target_link_libraries(sphere_sampling ${OPENGL_LIBRARIES})
#target_link_libraries(sphere_sampling glfw ${GLFW_LIBRARIES})
#target_link_libraries(sphere_sampling ${GLEW_LIBRARIES})
target_link_libraries(ss ${ASSIMP_LIBRARIES})
#target_link_libraries(sphere_sampling ss)
