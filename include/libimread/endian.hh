
#ifndef LIBIMREAD_ENDIAN_HH_
#define LIBIMREAD_ENDIAN_HH_

#include <utility>

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
        
        template <typename T, typename U> inline
        T reconsider_cast(U uvula)  { return *((T*)&uvula); }
        
        template <typename T, typename U> inline
        T reconsider_cast(U* uvula) { return reconsider_cast<T, U>(*uvula); }
        
        __attribute__((__always_inline__))
        inline bool littleendian(void) {
            #if defined(__BIG_ENDIAN__)
                return false;
            #elif defined(__LITTLE_ENDIAN__)
                return true;
            #else
                /// Otherwise, do something quick to compute it:
                int i = 1;
                return reconsider_cast<char>(i);
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
        void swap_endian(T* f, int length = 1) {
            using std::swap;
            for (char* c = static_cast<char*>(f); length--; c += sizeof(T)) {
                switch (sizeof(T)) {
                    case 2:
                        swap(c[0], c[1]);
                        break;
                    case 4:
                        swap(c[0], c[3]);
                        swap(c[1], c[2]);
                        break;
                    case 8:
                        swap(c[0], c[7]);
                        swap(c[1], c[6]);
                        swap(c[2], c[5]);
                        swap(c[3], c[4]);
                        break;
                    case 16:
                    case 32:
                    case 64:
                        /// at these sizes, the loop overhead is worth it:
                        int half = sizeof(T) / 2,
                            almosthalf = sizeof(T) / 2 - 1;
                        for (int idx = 0; idx < half; ++idx) {
                            swap(c[idx], c[almosthalf - idx]);
                        }
                        break;
                }
            }
        }
    
    } /* namesapce detail */
    
} /* namespace im */

#ifndef SWAP_ENDIAN16
#define SWAP_ENDIAN16(value) ((value) = (((value) & 0xff) << 8) | (((value) & 0xff00) >> 8))
#endif /// SWAP_ENDIAN16

#endif /// LIBIMREAD_ENDIAN_HH_