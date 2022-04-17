cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(npt-mtk npt-mtk.cpp)

add_library(libextxyz SHARED IMPORTED)
set_property(TARGET libextxyz PROPERTY IMPORTED_LOCATION "/usr/local/lib/libextxyz.dylib")

add_library(libcleri SHARED IMPORTED)
set_property(TARGET libcleri PROPERTY IMPORTED_LOCATION "/usr/local/lib/libcleri.dylib")

target_link_libraries(npt-mtk "${TORCH_LIBRARIES}" libextxyz libcleri)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib -fno-exceptions -fopencilk")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -I/usr/local/include -fno-exceptions -fopencilk")
set_property(TARGET npt-mtk PROPERTY CXX_STANDARD 17)
