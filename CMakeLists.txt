cmake_minimum_required(VERSION 3.16..3.21)

# Set app name
set(APP_NAME FaceMeshCpp)

# Set 3rd party path
set(TFLite_PATH "C:/tensorflowlite")
set(TFLite_INCLUDE_DIRS "${TFLite_PATH}/include")
set(TFLite_LIBS "${TFLite_PATH}/libs/windows_x86_64/libtensorflowlite.so.if.lib")

# Set project
project(${APP_NAME})

# Make executable app.
add_executable(${APP_NAME})

# Add source file
add_subdirectory(src)

# Find opengl libraries
find_package(OpenCV REQUIRED)

# Add include path
target_include_directories(${APP_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} 
    PRIVATE ${TFLite_INCLUDE_DIRS})

# Link libraries to app.
target_link_libraries(${APP_NAME} 
    PRIVATE ${OpenCV_LIBS} 
    PRIVATE ${TFLite_LIBS}
)

# Build in multi-process.
target_compile_options(${APP_NAME} 
    PRIVATE /MP)