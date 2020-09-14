cmake_minimum_required(VERSION 3.5.1)

project(face-ds)
enable_language(CXX)
include(FindCUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${CUDA_INCLUDE_DIRS}
        /opt/nvidia/deepstream/deepstream-5.0/sources/includes
        /usr/include/glib-2.0
        /usr/include/gstreamer-1.0
)


add_executable(
        face-ds
        deepstream_infer_tensor_meta_test.cpp
        FaceDetector.h
        FaceDetector.cpp
)


target_link_libraries(
        face-ds
        ${CUDA_LIBRARIES}
        nvinfer
        nvinfer_plugin
        ${OpenCV_LIBS}
        -L/usr/local/cuda-10.2/lib64/ -lcudart
        -L/opt/nvidia/deepstream/deepstream-5.0/lib -Wl,-rpath=/opt/nvidia/deepstream/deepstream-5.0/lib
        -lnvdsgst_meta -lnvds_meta -lnvds_inferutils -lnvds_utils -lm -lnvds_infercustomparser
        -lgstreamer-1.0
        -lgobject-2.0
        -lglib-2.0
)
