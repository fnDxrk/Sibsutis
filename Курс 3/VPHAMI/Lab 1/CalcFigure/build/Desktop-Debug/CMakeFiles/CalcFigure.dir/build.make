# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug"

# Include any dependencies generated for this target.
include CMakeFiles/CalcFigure.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CalcFigure.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CalcFigure.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CalcFigure.dir/flags.make

CalcFigure_autogen/timestamp: /usr/lib/qt6/moc
CalcFigure_autogen/timestamp: /usr/lib/qt6/uic
CalcFigure_autogen/timestamp: CMakeFiles/CalcFigure.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir="/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Automatic MOC and UIC for target CalcFigure"
	/usr/bin/cmake -E cmake_autogen "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles/CalcFigure_autogen.dir/AutogenInfo.json" Debug
	/usr/bin/cmake -E touch "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CalcFigure_autogen/timestamp"

CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o: CMakeFiles/CalcFigure.dir/flags.make
CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o: CalcFigure_autogen/mocs_compilation.cpp
CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o: CMakeFiles/CalcFigure.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o -MF CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o.d -o CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o -c "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CalcFigure_autogen/mocs_compilation.cpp"

CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CalcFigure_autogen/mocs_compilation.cpp" > CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.i

CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CalcFigure_autogen/mocs_compilation.cpp" -o CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.s

CMakeFiles/CalcFigure.dir/main.cpp.o: CMakeFiles/CalcFigure.dir/flags.make
CMakeFiles/CalcFigure.dir/main.cpp.o: /home/dxrk_/Documents/Sibsutis/Курс\ 3/VPHAMI/Lab\ 1/CalcFigure/main.cpp
CMakeFiles/CalcFigure.dir/main.cpp.o: CMakeFiles/CalcFigure.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/CalcFigure.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CalcFigure.dir/main.cpp.o -MF CMakeFiles/CalcFigure.dir/main.cpp.o.d -o CMakeFiles/CalcFigure.dir/main.cpp.o -c "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/main.cpp"

CMakeFiles/CalcFigure.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CalcFigure.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/main.cpp" > CMakeFiles/CalcFigure.dir/main.cpp.i

CMakeFiles/CalcFigure.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CalcFigure.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/main.cpp" -o CMakeFiles/CalcFigure.dir/main.cpp.s

CMakeFiles/CalcFigure.dir/mainwindow.cpp.o: CMakeFiles/CalcFigure.dir/flags.make
CMakeFiles/CalcFigure.dir/mainwindow.cpp.o: /home/dxrk_/Documents/Sibsutis/Курс\ 3/VPHAMI/Lab\ 1/CalcFigure/mainwindow.cpp
CMakeFiles/CalcFigure.dir/mainwindow.cpp.o: CMakeFiles/CalcFigure.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/CalcFigure.dir/mainwindow.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CalcFigure.dir/mainwindow.cpp.o -MF CMakeFiles/CalcFigure.dir/mainwindow.cpp.o.d -o CMakeFiles/CalcFigure.dir/mainwindow.cpp.o -c "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/mainwindow.cpp"

CMakeFiles/CalcFigure.dir/mainwindow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CalcFigure.dir/mainwindow.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/mainwindow.cpp" > CMakeFiles/CalcFigure.dir/mainwindow.cpp.i

CMakeFiles/CalcFigure.dir/mainwindow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CalcFigure.dir/mainwindow.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/mainwindow.cpp" -o CMakeFiles/CalcFigure.dir/mainwindow.cpp.s

# Object files for target CalcFigure
CalcFigure_OBJECTS = \
"CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o" \
"CMakeFiles/CalcFigure.dir/main.cpp.o" \
"CMakeFiles/CalcFigure.dir/mainwindow.cpp.o"

# External object files for target CalcFigure
CalcFigure_EXTERNAL_OBJECTS =

CalcFigure: CMakeFiles/CalcFigure.dir/CalcFigure_autogen/mocs_compilation.cpp.o
CalcFigure: CMakeFiles/CalcFigure.dir/main.cpp.o
CalcFigure: CMakeFiles/CalcFigure.dir/mainwindow.cpp.o
CalcFigure: CMakeFiles/CalcFigure.dir/build.make
CalcFigure: /usr/lib/libQt6Widgets.so.6.7.2
CalcFigure: /usr/lib/libQt6Gui.so.6.7.2
CalcFigure: /usr/lib/libQt6Core.so.6.7.2
CalcFigure: /usr/lib/libGLX.so
CalcFigure: /usr/lib/libOpenGL.so
CalcFigure: CMakeFiles/CalcFigure.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable CalcFigure"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CalcFigure.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CalcFigure.dir/build: CalcFigure
.PHONY : CMakeFiles/CalcFigure.dir/build

CMakeFiles/CalcFigure.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CalcFigure.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CalcFigure.dir/clean

CMakeFiles/CalcFigure.dir/depend: CalcFigure_autogen/timestamp
	cd "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure" "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure" "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug" "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug" "/home/dxrk_/Documents/Sibsutis/Курс 3/VPHAMI/Lab 1/CalcFigure/build/Desktop-Debug/CMakeFiles/CalcFigure.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/CalcFigure.dir/depend

