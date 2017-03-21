
#ifndef LIBIMREAD_TESTS_HELPERS_COLLECT_HH_
#define LIBIMREAD_TESTS_HELPERS_COLLECT_HH_

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>

#ifndef COLLECT_TEMPORARIES
#define COLLECT_TEMPORARIES 0
#endif

#ifndef CHECK_DIRECTORY
#define CHECK_DIRECTORY "/Users/fish/Dropbox/libimread/check/"
#endif

template <typename P>
bool COLLECT(P&& p) {
    using filesystem::path;
    #if COLLECT_TEMPORARIES == 1
        path::makedir(CHECK_DIRECTORY);
        return path(std::forward<P>(p)).rename(CHECK_DIRECTORY);
    #else
        return path::remove(std::forward<P>(p));
    #endif
}

#endif /// LIBIMREAD_TESTS_HELPERS_COLLECT_HH_