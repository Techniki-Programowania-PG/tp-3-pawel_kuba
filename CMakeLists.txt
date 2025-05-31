# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Dodaj Matplot++ jako subdirectory (zakładam, że jest w external/matplotplusplus)
add_subdirectory(external/matplotplusplus)

# Twórz bibliotekę Pythonową
python_add_library(_core MODULE src/main.cpp WITH_SOABI)

# Linkuj pybind11 headers i matplot
target_link_libraries(_core PRIVATE pybind11::headers matplot)

# Dodaj include dirs Matplot++ (jeśli add_subdirectory tego nie zrobił, to dodaj ręcznie)
target_include_directories(_core PRIVATE external/matplotplusplus/include)

# Definicje preprocesora
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Instalacja
install(TARGETS _core DESTINATION scikit_build_example)
