# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/oleg/cmake-3.14.5-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/oleg/cmake-3.14.5-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/3D_object_tracking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/3D_object_tracking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/3D_object_tracking.dir/flags.make

CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o: CMakeFiles/3D_object_tracking.dir/flags.make
CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o: ../src/camFusion_Student.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o"
	/bin/clang++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o -c /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/camFusion_Student.cpp

CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.i"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/camFusion_Student.cpp > CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.i

CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.s"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/camFusion_Student.cpp -o CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.s

CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o: CMakeFiles/3D_object_tracking.dir/flags.make
CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o: ../src/FinalProject_Camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o"
	/bin/clang++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o -c /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/FinalProject_Camera.cpp

CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.i"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/FinalProject_Camera.cpp > CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.i

CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.s"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/FinalProject_Camera.cpp -o CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.s

CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o: CMakeFiles/3D_object_tracking.dir/flags.make
CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o: ../src/lidarData.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o"
	/bin/clang++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o -c /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/lidarData.cpp

CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.i"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/lidarData.cpp > CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.i

CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.s"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/lidarData.cpp -o CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.s

CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o: CMakeFiles/3D_object_tracking.dir/flags.make
CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o: ../src/matching2D_Student.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o"
	/bin/clang++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o -c /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/matching2D_Student.cpp

CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.i"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/matching2D_Student.cpp > CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.i

CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.s"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/matching2D_Student.cpp -o CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.s

CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o: CMakeFiles/3D_object_tracking.dir/flags.make
CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o: ../src/objectDetection2D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o"
	/bin/clang++-8  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o -c /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/objectDetection2D.cpp

CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.i"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/objectDetection2D.cpp > CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.i

CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.s"
	/bin/clang++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/src/objectDetection2D.cpp -o CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.s

# Object files for target 3D_object_tracking
3D_object_tracking_OBJECTS = \
"CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o" \
"CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o" \
"CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o" \
"CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o" \
"CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o"

# External object files for target 3D_object_tracking
3D_object_tracking_EXTERNAL_OBJECTS =

3D_object_tracking: CMakeFiles/3D_object_tracking.dir/src/camFusion_Student.cpp.o
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/src/FinalProject_Camera.cpp.o
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/src/lidarData.cpp.o
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/src/matching2D_Student.cpp.o
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/src/objectDetection2D.cpp.o
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/build.make
3D_object_tracking: /usr/local/lib/libopencv_gapi.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_stitching.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_aruco.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_bgsegm.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_bioinspired.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_ccalib.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_dnn_objdetect.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_dpm.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_face.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_freetype.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_fuzzy.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_hdf.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_hfs.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_img_hash.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_line_descriptor.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_quality.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_reg.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_rgbd.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_saliency.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_stereo.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_structured_light.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_superres.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_surface_matching.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_tracking.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_videostab.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_viz.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_xfeatures2d.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_xobjdetect.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_xphoto.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_shape.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_datasets.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_plot.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_text.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_dnn.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_highgui.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_ml.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_phase_unwrapping.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_optflow.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_ximgproc.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_video.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_videoio.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_imgcodecs.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_objdetect.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_calib3d.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_features2d.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_flann.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_photo.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_imgproc.so.4.1.1
3D_object_tracking: /usr/local/lib/libopencv_core.so.4.1.1
3D_object_tracking: CMakeFiles/3D_object_tracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable 3D_object_tracking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/3D_object_tracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/3D_object_tracking.dir/build: 3D_object_tracking

.PHONY : CMakeFiles/3D_object_tracking.dir/build

CMakeFiles/3D_object_tracking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/3D_object_tracking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/3D_object_tracking.dir/clean

CMakeFiles/3D_object_tracking.dir/depend:
	cd /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build /home/oleg/Documents/SensorFusion/SFND_3D_Object_Tracking/build/CMakeFiles/3D_object_tracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/3D_object_tracking.dir/depend

