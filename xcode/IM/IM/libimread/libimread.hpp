/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_CMAKE_HPP_IN_
#define LIBIMREAD_CMAKE_HPP_IN_

#include <cstdint>

#define IM_VERSION_MAJOR                0
#define IM_VERSION_MINOR                0
#define IM_VERSION_PATCH                0
#define IM_VERSION                      "0.0.0"

#define IM_COLOR_TRACE                  1
#define IM_VERBOSE                      0

#define FILESYSTEM_TEMP_SUFFIX          ".imdata"
#define FILESYSTEM_TEMP_FILENAME        "temporary-XXXXXXXX"
#define FILESYSTEM_TEMP_DIRECTORYNAME   "libimread-XXXXXXX"

namespace im {
    
    typedef uint8_t byte;
    
    /// unpack {
    ///     (out += std::to_string(array[I]))...
    /// };
    #define unpack                     __attribute__((unused)) int unpacker[]
    
    /// virtual void func() warn_unused = 0;
    #define warn_unused                __attribute__((__warn_unused_result__))
    
    /*
    #define alias(source)              __attribute__((weak, alias ( #source )))
    #define required(idx, ...)         __attribute__((__nonnull__(idx, ##__VA_ARGS__)))
    #define unused                     __attribute__((__unused__))
    #define always_inline              __attribute__((__always_inline__))
    #define overloadable               __attribute__((__overloadable__))
    #define always_const               __attribute__((__const__))
    #define always_pure                __attribute__((__pure__))
    */
     
}

#endif /// LIBIMREAD_CMAKE_HPP_IN_
