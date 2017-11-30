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
find_package_handle_standard_args(${lib_name}
                                  DEFAULT_MSG
                                  ${lib_name}_include_dir
                                  ${lib_name}_FOUND)

# Mark variables 'advanced'
mark_as_advanced(${lib_name}_FOUND
                 ${lib_name}_include_dir
                 ${lib_name})

set(${lib_name}_include_dir ${CMAKE_SOURCE_DIR}/include/libimread PARENT_SCOPE)

# Include custom CMake macro
include(macro)

# Include `check_include_file()` macro/function and friends
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckTypeSize)
include(CheckFunctionExists)
include(CheckDevEntry)

# Check for architecture-dependent headers:
check_include_file(fcntl.h                              HAVE_FCNTL_H)
check_include_file(dirent.h                             HAVE_DIRENT_H)
check_include_file(unistd.h                             HAVE_UNISTD_H)
check_include_file(dlfcn.h                              HAVE_DLFCN_H)
check_include_file(pthread.h                            HAVE_PTHREAD_H)
check_include_file(utime.h                              HAVE_UTIME_H)
check_include_file(pwd.h                                HAVE_PWD_H)
check_include_file(glob.h                               HAVE_GLOB_H)
check_include_file(wordexp.h                            HAVE_WORDEXP_H)

# Check for (processor-dependant) intrinsics headers:
check_include_file(immintrin.h                          HAVE_IMMINTRIN_H)
check_include_file(x86intrin.h                          HAVE_X86INTRIN_H)

# Check for (possibly non-standard) C++ headers:
check_include_file_cxx(cxxabi.h                         HAVE_CXXABI_HH)
check_include_file_cxx(experimental/string_view         HAVE_EXPERIMENTAL_STRINGVIEW_HH)
check_include_file_cxx(string_view                      HAVE_STRINGVIEW_HH)
check_include_file_cxx(cfenv                            HAVE_CFENV_HH)

# Check for operating-system-dependent headers:
check_include_file(sys/ioctl.h                          HAVE_SYS_IOCTL_H)
check_include_file(sys/sysctl.h                         HAVE_SYS_SYSCTL_H)
check_include_file(sys/types.h                          HAVE_SYS_TYPES_H)
check_include_file(sys/stat.h                           HAVE_SYS_STAT_H)
check_include_file(sys/time.h                           HAVE_SYS_TIME_H)
check_include_file(sys/mman.h                           HAVE_SYS_MMAN_H)
check_include_file(sys/resource.h                       HAVE_SYS_RESOURCE_H)
check_include_file(sys/socket.h                         HAVE_SYS_SOCKET_H)
check_include_file(sys/sendfile.h                       HAVE_SYS_SENDFILE_H)
check_include_file(sys/uio.h                            HAVE_SYS_UIO_H)
check_include_file(sys/extattr.h                        HAVE_SYS_EXTATTR_H)
check_include_file(sys/xattr.h                          HAVE_SYS_XATTR_H)
check_include_file(netinet/in.h                         HAVE_NETINET_IN_H)

# Check for headers known to be non-standard:
check_include_file(alloca.h                             HAVE_ALLOCA_H)
# check_include_file(config.h                             HAVE_CONFIG_H)
check_include_file(getopt.h                             HAVE_GETOPT_H)


if(APPLE)
    
    # Check for headers specific to Darwin and OS X:
    check_include_file(copyfile.h                       HAVE_COPYFILE_H)
    check_include_file(mach-o/dyld.h                    HAVE_MACHO_DYLD_H)
    check_include_file(objc/message.h                   HAVE_OBJC_MESSAGE_H)
    check_include_file(objc/runtime.h                   HAVE_OBJC_RUNTIME_H)
    check_include_file(libkern/OSTypes.h                HAVE_LIBKERN_OSTYPES_H)
    
    # Check for /dev entries (only on OS X for now):
    check_dev_entry("autofs_nowait"                     HAVE_AUTOFS_NOWAIT)
    check_dev_entry("autofs_notrigger"                  HAVE_AUTOFS_NOTRIGGER)
    
else(APPLE)
    
    # Set all results to false for checks related to OS X:
    set(HAVE_COPYFILE_H NO
        CACHE INTERNAL
        "<copyfile.h> presence"
        FORCE)
    
    set(HAVE_MACHO_DYLD_H NO
        CACHE INTERNAL
        "<mach-o/dyld.h> presence"
        FORCE)
    
    set(HAVE_OBJC_MESSAGE_H NO
        CACHE INTERNAL
        "<objc/message.h> presence"
        FORCE)
    
    set(HAVE_OBJC_RUNTIME_H NO
        CACHE INTERNAL
        "<objc/runtime.h> presence"
        FORCE)
    
    set(HAVE_LIBKERN_OSTYPES_H NO
        CACHE INTERNAL
        "<libkern/OSTypes.h> presence"
        FORCE)
    
    set(HAVE_AUTOFS_NOWAIT NO
        CACHE INTERNAL
        "/dev/autofs_nowait presence"
        FORCE)
    
    set(HAVE_AUTOFS_NOTRIGGER NO
        CACHE INTERNAL
        "/dev/autofs_notrigger presence"
        FORCE)
    
endif(APPLE)

# Check for a few (possibly non-standard) POSIX-ish functions:
set(CMAKE_REQUIRED_INCLUDES limits.h stdlib.h)
set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_function_exists(canonicalize_file_name            HAVE_CANONICALIZE_FILE_NAME_F)
check_function_exists(realpath                          HAVE_REALPATH_F)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_DEFINITIONS)

# Figure out the deal with 128-bit integer types:
set(CMAKE_EXTRA_INCLUDE_FILES stdint.h)
check_type_size("__int128_t"                            INT128_T)
check_type_size("__uint128_t"                           UINT128_T)
set(CMAKE_EXTRA_INCLUDE_FILES)

if(NOT ${INT128_T})
    message(WARNING "__int128_t not found")
endif(NOT ${INT128_T})
if(NOT ${UINT128_T})
    message(WARNING "__uint128_t not found")
endif(NOT ${UINT128_T})
