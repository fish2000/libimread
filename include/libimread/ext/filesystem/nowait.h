/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_
#define LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_

#include <atomic>
#include <libimread/libimread.hpp>

/// This header, nowait.h, and its corresponding implementation file, nowait.cpp,
/// furnish RAII structures implementing an idiom found in Apple’s published
/// open-source codebase for the CoreFoundation framework (q.v. the GitHub repo
/// https://github.com/opensource-apple/CF sub.) in which acquiring a descriptor
/// on a /dev filesystem entry before issuing POSIX calls (and subsequently closing
/// that descriptor afterward) will enable certain filesystem-level optimizations --
/// optimizations that may have side-effects and are therefore not universally enabled
/// by default. 
///
/// The RAII structures are themselves implemented with convenience macros, as they
/// each operate identically, aside from the specific /dev entry each RAII structure
/// effectively manages. To create a new structure of this sort, use the macro
/// VFS_INHIBITOR_DECLARATION(…) in your header file, and VFS_INHIBITOR_DEFINITION(…)
/// in the corresponding implementation -- for example, to manage descriptors on
/// a new /dev entry /dev/autofs_yodogg, you would do it like this:
///
///     /* yodogg.hh */
///     namespace iheardyoulike { /// or what have you
///         VFS_INHIBITOR_DECLARATION(yodogg_t);
///     }
/// 
///     /* yodogg.cc */
///     namespace iheardyoulike { /// or whichever you used in the header file
///         VFS_INHIBITOR_DEFINITION(yodogg_t, "/dev/autofs_yodogg");
///     }
///
/// … this will yield you a `iheardyoulike::yodogg_t` struct, which you can use like so:
///
///     {
///         iheardyoulike::yodogg_t yodogg;
///         relevant_POSIX_call();
///         another_relevant_POSIX_call();
///     }
///
/// … which will execute those relevant POSIX calls while the /dev descriptor is held,
/// and automatically release the descriptor on scope exit -- which, in the snippet above,
/// is when the curly braces close out.
/// 
/// I use some cmake-defined preprocessor values to check for the existence of these
/// /dev entries during the pre-build, and fill in an empty struct using an additional
/// macro VFS_INHIBITOR_EMPTY_STRUCT(…) when the related /dev entries aren’t available.
/// This allows me to festoon my POSIX filesystem code with RAII dev-entry-management
/// stuff as I please, knowing that they will work as advertised on an Apple platform,
/// and compile down to NOOPs anywhere else.
///
/// Internally, each RAII structure uses std::atomic<…> values to track its descriptor
/// and retain-count values to guarantee thread-safe operation. This will, of course,
/// require C++11 compilation at least… but you can totally handle that, right? Right.
///
/// My RAII structures are based on the CoreFoundation c99 code here:
///     https://git.io/vQYXO
/// Use of the original Apple idiom within CoreFoundation’s POSIX interface here:
///     https://github.com/opensource-apple/CF/blob/master/CFFileUtilities.c
/// Background information on the subject is here:
///     https://stackoverflow.com/q/39403803/298171

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
                if (retaincount.fetch_add(1) == 0) {                    \
                    descriptor.store(::open(__device__, 0));            \
                }                                                       \
            }                                                           \
                                                                        \
            __typename__::~__typename__() {                             \
                if (retaincount.fetch_sub(1) == 1) {                    \
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