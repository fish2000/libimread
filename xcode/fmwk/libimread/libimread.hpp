/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_CMAKE_HPP_IN_
#define LIBIMREAD_CMAKE_HPP_IN_

#include <cstdint>
#include <vector>

#ifndef __has_feature                       // Optional.
#define __has_feature(x)                0   // Compatibility with non-clang compilers.
#endif

#define IM_VERSION_MAJOR                0
#define IM_VERSION_MINOR                0
#define IM_VERSION_PATCH                0
#define IM_VERSION                      "0.0.0"

#define IM_INSTALL_PREFIX               "/usr/local"
#define IM_COMPILE_OPTIONS              ""
#define IM_INCLUDE_DIRECTORIES          "/Users/fish/Dropbox/libimread/include/libimread;/Users/fish/Dropbox/libimread/xcode/fmwk/../include;/Users/fish/Dropbox/libimread/xcode/fmwk;/Users/fish/Dropbox/libimread/deps/crossguid;/Users/fish/Dropbox/libimread/deps/docopt;/Users/fish/Dropbox/libimread/deps/MABlockClosure;/Users/fish/Dropbox/libimread/deps/iod;/Users/fish/Dropbox/libimread/deps/SG14;/Users/fish/Dropbox/libimread/deps/SSZipArchive;/usr/local/include;/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include;/usr/local/include;/usr/local/include;/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include;/usr/local/include;/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include;/usr/local/Cellar/hdf5/1.8.16_1/include"
#define IM_LINK_LIBRARIES               "IM_LINK_LIBRARIES-NOTFOUND"
#define IM_LINK_FLAGS                   "-m64"

#define IM_COLOR_TRACE                  1
#define IM_VERBOSE                      0

#define FILESYSTEM_TEMP_SUFFIX          ".imdata"
#define FILESYSTEM_TEMP_FILENAME        "temporary-XXXXXXXX"
#define FILESYSTEM_TEMP_DIRECTORYNAME   "libimread-XXXXXXX"

namespace im {
    
    /// `byte` is a way better token than `uint8_t` from, like,
    /// a human mnemonic perspective.
    
    using byte = uint8_t;
    
    /// `unpack` is a sleazy trick for -- YOU GUESSED IT -- 
    /// expanding variadic template parameter packs. USAGE:
    ///
    ///     unpack {
    ///         (out += std::to_string(array[I]))...
    ///     };
    
    #define unpack                     __attribute__((unused)) int unpacker[]
    
    /// Use `warn_unused` like a trailing function-definition keyword:
    ///
    ///     void yodogg(std::string const& s) const warn_unused { … }
    
    #define warn_unused                __attribute__((__warn_unused_result__))
    
    /// The following fake function-definition keywords may be of use
    /// to the future generations of programmers who will serve this 
    /// great and mighty codebase (personally I could not give AF
    /// and I think macros are low-class):
    
    // #define alias(source)           __attribute__((weak, alias ( #source )))
    // #define required(idx, ...)      __attribute__((__nonnull__(idx, ##__VA_ARGS__)))
    // #define unused                  __attribute__((__unused__))
    // #define always_inline           __attribute__((__always_inline__))
    // #define overloadable            __attribute__((__overloadable__))
    // #define always_const            __attribute__((__const__))
    // #define always_pure             __attribute__((__pure__))
     
}

#endif /// LIBIMREAD_CMAKE_HPP_IN_
