# Author: Félix C. Morency
# 2011.10

#Set the following variable
# halide_FOUND: True if the library has been found
# halide_include_dir: Path to header files
include(CheckIncludeFiles)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(FindPackageHandleStandardArgs)

# find_package(PkgConfig)
# pkg_check_modules(HALIDE Halide)

if (NOT(HALIDE_FOUND))
    # set(HALIDE_FOUND_HEADER NO)
    # set(HALIDERUNTIME_FOUND_HEADER NO)
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_REQUIRED_INCLUDES
        "/usr/local/opt/halide/include"
        "/usr/local/include"
        "/opt/local/include"
        "~/.local/include")
    
    # check_include_file_cxx(
    #     Halide.h
    #     HALIDE_FOUND_HEADER)
    
    check_include_file_cxx(
        Halide.h
        HALIDE_FOUND_HEADER)
    
    # if (NOT HALIDE_FOUND_HEADER)
    #     check_include_file(
    #         Halide.h
    #         HALIDE_FOUND_HEADER)
    # endif(HALIDE_FOUND_HEADER)
    
    if (NOT HALIDE_FOUND_HEADER)
        check_include_file_cxx(
            HalideRuntime.h
            HALIDERUNTIME_FOUND_HEADER)
    endif(HALIDE_FOUND_HEADER)
    
    # if-else mess is if-elsey (I know)
    if (HALIDERUNTIME_FOUND_HEADER)
        # NOOp
    else(HALIDERUNTIME_FOUND_HEADER)
        if (HALIDE_FOUND_HEADER)
            # NOOp
        else(HALIDE_FOUND_HEADER)
            message(FATAL_ERROR "** Can't find Halide.h or HalideRuntime.h")
        endif(HALIDE_FOUND_HEADER)
        
    endif(HALIDERUNTIME_FOUND_HEADER)
    
endif()

set(HALIDE_LIBRARY "-lHalide")
set(HALIDE_FOUND TRUE)
set(HALIDE_INCLUDE_DIR HALIDE_FOUND_HEADER)

# OMG CMAKE SOMETIMES YOURE LIKE IF FUCKING ANDY KAUFMAN WROTE INTERCAL I SWEAR TO FUCKING CHRIST
# SRSLY U GUYS THIS NEXT BIT IS TOTALLY BEST FUCKING PRACTICES IN LIKE CMAKES OWN STANDARD LIBRARY
set(HALIDE_LIBRARIES ${HALIDE_LIBRARY})
set(HALIDE_INCLUDE_DIRS ${HALIDE_INCLUDE_DIR})

# and then this one lets you bar mitzvah your variables -- I have no idea what this does:
mark_as_advanced(
    HALIDE_LIBRARY      HALIDE_INCLUDE_DIR
    HALIDE_LIBRARIES    HALIDE_INCLUDE_DIRS)

find_package_handle_standard_args(HALIDE
    DEFAULT_MSG
    HALIDE_LIBRARY HALIDE_INCLUDE_DIR)

