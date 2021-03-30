cmake_minimum_required(VERSION 3.13)

project(PersonTracking)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
HelpEntities.cpp
HungarianAlgorithm.cpp
KalmanTracker.cpp
ObjDetect.cpp
ObjTracker.cpp
)

set(HEADER_FILES
HungarianAlgorithm.h
KalmanTracker.h
ObjDetect.h
ObjTracker.h
)

add_executable(PersonTracking 
main.cpp
${SOURCE_FILES}
${HEADER_FILES}
)