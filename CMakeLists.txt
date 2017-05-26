cmake_minimum_required(VERSION 3.5.1)
project(sphere_sampling)

#c++ 11
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.9")

#for autocomplete database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#find_package(glfw3 3.2 REQUIRED)
#find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)
find_package(Assimp REQUIRED)
find_package(CUDA REQUIRED)

#CUDA OPTIONS
set(CUDA_64_BIT_DEVICE_CODE ON)
set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 --use_fast_math -Xptxas -O3 --maxrregcount 40") 

message(${CUDA_NVCC_FLAGS})

include_directories(./include)

include_directories(/usr/local/cuda/include/)
include_directories(/usr/local/cuda/samples/common/inc)

CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

set(CUDA_SOURCES  src/boundary.cu)
CUDA_COMPILE(CUDA_OBJ ${CUDA_SOURCES}) 

add_library(ss src/ss.cpp src/boundary_forces.cpp ${CUDA_OBJ})

#add_executable(sphere_sampling main.cpp)

##target_include_directories(sph ${OPENGL_INCLUDE_DIR})
#target_link_libraries(sphere_sampling ${OPENGL_gl_LIBRARY})
#target_link_libraries(sphere_sampling ${OPENGL_LIBRARIES})
#target_link_libraries(sphere_sampling glfw ${GLFW_LIBRARIES})
#target_link_libraries(sphere_sampling ${GLEW_LIBRARIES})
target_link_libraries(ss ${ASSIMP_LIBRARIES})
#target_link_libraries(sphere_sampling ss)
