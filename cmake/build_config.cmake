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

# Include `check_include_file()` macro/function and friends
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckTypeSize)

# Check required headers
check_include_file(fcntl.h              HAVE_FCNTL_H)
check_include_file(dirent.h             HAVE_DIRENT_H)
check_include_file(unistd.h             HAVE_UNISTD_H)
# check_include_file(dlfcn.h            HAVE_DLFCN_H)
# check_include_file(pthread.h          HAVE_PTHREAD_H)
check_include_file(pwd.h                HAVE_PWD_H)
check_include_file(glob.h               HAVE_GLOB_H)
check_include_file_cxx(cxxabi.h         HAVE_CXXABI_H)

# check_include_file(sys/ioctl.h        HAVE_SYS_IOCTL_H)
check_include_file(sys/types.h          HAVE_SYS_TYPES_H)
check_include_file(sys/stat.h           HAVE_SYS_STAT_H)
check_include_file(sys/time.h           HAVE_SYS_TIME_H)

if(APPLE)
    check_include_file(mach-o/dyld.h    HAVE_MACHO_DYLD_H)
    check_include_file(objc/message.h   HAVE_OBJC_MESSAGE_H)
    check_include_file(objc/runtime.h   HAVE_OBJC_RUNTIME_H)
    check_include_file(sys/mman.h       HAVE_SYS_MMAN_H)
endif(APPLE)

set(CMAKE_EXTRA_INCLUDE_FILES "stdint.h")
check_type_size("__int128_t"  INT128_T)
check_type_size("__uint128_t" UINT128_T)
set(CMAKE_EXTRA_INCLUDE_FILES)

if(NOT ${INT128_T})
    message(WARNING "__int128_t not found")
endif(NOT ${INT128_T})
if(NOT ${UINT128_T})
    message(WARNING "__uint128_t not found")
endif(NOT ${UINT128_T})
