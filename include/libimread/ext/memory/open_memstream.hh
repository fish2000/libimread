#ifndef LIBIMREAD_EXT_OPEN_MEMSTREAM_HH_
#define LIBIMREAD_EXT_OPEN_MEMSTREAM_HH_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <libimread/libimread.hpp>

namespace memory {
    
    struct memstream {
        char** cp;
        std::size_t* lenp;
        std::size_t offset;
    };
    
    typedef struct memstream mstream;
}

FILE* open_memstream(char** cp, std::size_t* lenp);

#endif /// LIBIMREAD_EXT_OPEN_MEMSTREAM_HH_
