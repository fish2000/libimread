#ifndef OPEN_MEMSTREAM_H_
#define OPEN_MEMSTREAM_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <libimread/libimread.hpp>

namespace memory {

    struct memstream {
        char **cp;
        std::size_t *lenp;
        std::size_t offset;
    };

    typedef struct memstream mstream;
}

FILE *open_memstream(char **cp, std::size_t *lenp);

#endif // #ifndef FMEMOPEN_H_
