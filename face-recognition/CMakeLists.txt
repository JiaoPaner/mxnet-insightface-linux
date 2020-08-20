cmake_minimum_required(VERSION 3.16)
project(face_recognition)

set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_C_FLAGS "-DHAVE_NETINET_IN_H")
string(APPEND CMAKE_C_FLAGS "-Wwrite-strings")

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
link_directories(lib)

file(GLOB src src/*.cpp src/*.c src/*.hpp src/*.h)
set(SRC ${src})

add_executable(face_recognition ${SRC})
#add_library(face_recognition SHARED ${SRC})

target_link_libraries(face_recognition ${OpenCV_LIBS} mxnet)