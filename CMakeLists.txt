cmake_minimum_required(VERSION 3.0)
project(StereoToPointCloud)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 设置CMAKE_PREFIX_PATH以包含JsonCpp安装路径
set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/jsoncpp/" ${CMAKE_PREFIX_PATH})

# 找到所需的库
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JsonCpp REQUIRED jsoncpp)

# 包含头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${VTK_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${JsonCpp_INCLUDE_DIRS})

# 添加可执行文件
add_executable(stereo_PCL main.cpp
)

# 链接库
target_link_libraries(stereo_PCL ${OpenCV_LIBS} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${JsonCpp_LIBRARIES})
