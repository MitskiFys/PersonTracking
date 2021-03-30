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

SET(OPENCV_DIR $ENV{OPENCV_DIR})
SET(OPENCV_INCLUDE_DIR ${OPENCV_DIR}/../../include)
include_directories(${OPENCV_INCLUDE_DIR})
link_directories(${OPENCV_DIR}/lib)

add_executable(PersonTracking 
main.cpp
${SOURCE_FILES}
${HEADER_FILES}
)

target_link_libraries(
PersonTracking 
opencv_aruco451.lib
opencv_bgsegm451.lib
opencv_bioinspired451.lib
opencv_calib3d451.lib
opencv_ccalib451.lib
opencv_core451.lib
opencv_datasets451.lib
opencv_dnn_objdetect451.lib
opencv_dnn_superres451.lib
opencv_dnn451.lib
opencv_dpm451.lib
opencv_face451.lib
opencv_features2d451.lib
opencv_flann451.lib
opencv_fuzzy451.lib
opencv_gapi451.lib
opencv_hfs451.lib
opencv_highgui451.lib
opencv_img_hash451.lib
opencv_imgcodecs451.lib
opencv_imgproc451.lib
opencv_intensity_transform451.lib
opencv_line_descriptor451.lib
opencv_mcc451.lib
opencv_ml451.lib
opencv_objdetect451.lib
opencv_optflow451.lib
opencv_phase_unwrapping451.lib
opencv_photo451.lib
opencv_plot451.lib
opencv_quality451.lib
opencv_rapid451.lib
opencv_reg451.lib
opencv_rgbd451.lib
opencv_saliency451.lib
opencv_shape451.lib
opencv_stereo451.lib
opencv_stitching451.lib
opencv_structured_light451.lib
opencv_superres451.lib
opencv_surface_matching451.lib
opencv_text451.lib
opencv_tracking451.lib
opencv_video451.lib
opencv_videoio451.lib
opencv_videostab451.lib
opencv_xfeatures2d451.lib
opencv_ximgproc451.lib
opencv_xobjdetect451.lib
opencv_xphoto451.lib
)