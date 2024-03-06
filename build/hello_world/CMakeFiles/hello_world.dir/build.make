# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/CUDALearn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/CUDALearn/build

# Include any dependencies generated for this target.
include hello_world/CMakeFiles/hello_world.dir/depend.make

# Include the progress variables for this target.
include hello_world/CMakeFiles/hello_world.dir/progress.make

# Include the compile flags for this target's objects.
include hello_world/CMakeFiles/hello_world.dir/flags.make

hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o: hello_world/CMakeFiles/hello_world.dir/flags.make
hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o: ../hello_world/hello_world.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/CUDALearn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o"
	cd /root/workspace/CUDALearn/build/hello_world && /usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/workspace/CUDALearn/hello_world/hello_world.cu -o CMakeFiles/hello_world.dir/hello_world.cu.o

hello_world/CMakeFiles/hello_world.dir/hello_world.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hello_world.dir/hello_world.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

hello_world/CMakeFiles/hello_world.dir/hello_world.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hello_world.dir/hello_world.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target hello_world
hello_world_OBJECTS = \
"CMakeFiles/hello_world.dir/hello_world.cu.o"

# External object files for target hello_world
hello_world_EXTERNAL_OBJECTS =

hello_world/hello_world: hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o
hello_world/hello_world: hello_world/CMakeFiles/hello_world.dir/build.make
hello_world/hello_world: hello_world/CMakeFiles/hello_world.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/CUDALearn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable hello_world"
	cd /root/workspace/CUDALearn/build/hello_world && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_world.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hello_world/CMakeFiles/hello_world.dir/build: hello_world/hello_world

.PHONY : hello_world/CMakeFiles/hello_world.dir/build

hello_world/CMakeFiles/hello_world.dir/clean:
	cd /root/workspace/CUDALearn/build/hello_world && $(CMAKE_COMMAND) -P CMakeFiles/hello_world.dir/cmake_clean.cmake
.PHONY : hello_world/CMakeFiles/hello_world.dir/clean

hello_world/CMakeFiles/hello_world.dir/depend:
	cd /root/workspace/CUDALearn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/CUDALearn /root/workspace/CUDALearn/hello_world /root/workspace/CUDALearn/build /root/workspace/CUDALearn/build/hello_world /root/workspace/CUDALearn/build/hello_world/CMakeFiles/hello_world.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hello_world/CMakeFiles/hello_world.dir/depend
