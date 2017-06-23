/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_
#define LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_

#include <atomic>
#include <libimread/libimread.hpp>

namespace filesystem {
    
    namespace detail {
        
        #define VFS_INHIBITOR_DECLARATION(__typename__)                 \
            struct __typename__ final {                                 \
                                                                        \
                __typename__();                                         \
                ~__typename__();                                        \
                                                                        \
                private:                                                \
                    __typename__(__typename__ const&);                  \
                    __typename__(__typename__&&);                       \
                    __typename__& operator=(__typename__ const&);       \
                    __typename__& operator=(__typename__&&);            \
                                                                        \
                private:                                                \
                    static std::atomic<int> descriptor;                 \
                    static std::atomic<int> retaincount;                \
                                                                        \
            };
        
        #define VFS_INHIBITOR_EMPTY_STRUCT(__typename__)                \
            struct __typename__ {};
        
        #define VFS_INHIBITOR_DEFINITION(__typename__, __device__)      \
            std::atomic<int> __typename__::descriptor{ -1 };            \
            std::atomic<int> __typename__::retaincount{ 0 };            \
                                                                        \
            __typename__::__typename__() {                              \
                if (retaincount.fetch_add(1) == 1) {                    \
                    descriptor.store(::open(__device__, 0));            \
                }                                                       \
            }                                                           \
                                                                        \
            __typename__::~__typename__() {                             \
                if (retaincount.fetch_sub(1) == 0) {                    \
                    if (::close(descriptor.load()) == 0) {              \
                        descriptor.store(-1);                           \
                    }                                                   \
                }                                                       \
            }
        
        #ifdef IM_HAVE_AUTOFS_NOWAIT
        VFS_INHIBITOR_DECLARATION(nowait_t);
        #else
        VFS_INHIBITOR_EMPTY_STRUCT(nowait_t);
        #endif
        
        #ifdef IM_HAVE_AUTOFS_NOTRIGGER
        VFS_INHIBITOR_DECLARATION(notrigger_t);
        #else
        VFS_INHIBITOR_EMPTY_STRUCT(notrigger_t);
        #endif
        
    }
    
}

#endif /// LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_