cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(iot-application-26)

file(GLOB_RECURSE EI_SOURCES
    edge-impulse-sdk/*.cpp
    edge-impulse-sdk/*.cc
    edge-impulse-sdk/*.c
    tflite-model/*.cpp
    tflite-model/*.cc
    model-parameters/*.cpp
)

target_sources(app PRIVATE
    src/main.cpp
    ${EI_SOURCES}
)

target_include_directories(app PRIVATE
    .
    edge-impulse-sdk
    model-parameters
    tflite-model
)