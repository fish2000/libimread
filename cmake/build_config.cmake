# Author: Alexander Böhn (with Félix C. Morency)
# © 2011.10 -- GPL, Motherfuckers

# Path to the include directory
set(source_dir src)

# Set the library name and include directory
set(lib_name libimread)
set(${lib_name}_FOUND TRUE)
set(${lib_name}_include_dir ${CMAKE_SOURCE_DIR}/include/libimread)

# Indicate that we have 'found' the library as a package
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${lib_name} DEFAULT_MSG
    ${lib_name}_include_dir)

# Mark variables 'advanced'
mark_as_advanced(
    ${lib_name}_FOUND
    ${lib_name}_include_dir
    ${lib_name})

# Include custom CMake macro
include(macro)

