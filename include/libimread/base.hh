/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_BASE_HH_
#define LIBIMREAD_BASE_HH_

#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>
#include <libimread/image.hh>
#include <libimread/imagelist.hh>
#include <libimread/imageformat.hh>
#include <libimread/traits.hh>
#include <libimread/symbols.hh>
#include <libimread/options.hh>

namespace im {
    
    namespace detail {
        
        /// COURTESY OPENIMAGEIO:
        /// Try to deduce endianness
        #if (defined(_WIN32) || defined(__i386__) || defined(__x86_64__))
        #  ifndef __LITTLE_ENDIAN__
        #    define __LITTLE_ENDIAN__ 1
        #    undef __BIG_ENDIAN__
        #  endif
        #endif
        
        __attribute__((__always_inline__))
        inline bool littleendian(void) {
            #if defined(__BIG_ENDIAN__)
                return false;
            #elif defined(__LITTLE_ENDIAN__)
                return true;
            #else
                /// Otherwise, do something quick to compute it:
                int i = 1;
                return *((char*)&i);
            #endif
        }
        
        /// Return true if the architecture we are running on is big endian
        __attribute__((__always_inline__))
        inline bool bigendian(void) {
            return !littleendian();
        }
        
        /// COURTESY OPENIMAGEIO:
        /// Change endian-ness of one or more data items that are each 2, 4,
        /// or 8 bytes.  This should work for any of short, unsigned short, int,
        /// unsigned int, float, long long, pointers.
        template <typename T> inline
        void swap_endian(T* f, int len = 1) {
            using std::swap;
            for (char* c = (char*)f; len--; c += sizeof(T)) {
                if (sizeof(T) == 2) {
                    swap(c[0], c[1]);
                } else if (sizeof(T) == 4) {
                    swap(c[0], c[3]);
                    swap(c[1], c[2]);
                } else if (sizeof(T) == 8) {
                    swap(c[0], c[7]);
                    swap(c[1], c[6]);
                    swap(c[2], c[5]);
                    swap(c[3], c[4]);
                }
            }
        }
    
    } /* namesapce detail */
    
} /* namespace im */

#endif /// LIBIMREAD_BASE_HH_
