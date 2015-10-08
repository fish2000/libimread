/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_
#define LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_

#include <string>
#include <memory>
#include <mutex>
#include <functional>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <dirent.h>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    // enum class mode { READ, WRITE };
        
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        
        /// Deleter structures to close directory and file handles
        template <typename D>
        struct dircloser {
            constexpr dircloser() noexcept = default;
            template <typename U> dircloser(const dircloser<U>&) noexcept {};
            void operator()(D *dirhandle) { ::closedir(dirhandle); }
        };
    
        template <typename F>
        struct filecloser {
            constexpr filecloser() noexcept = default;
            template <typename U> filecloser(const filecloser<U>&) noexcept {};
            void operator()(F *filehandle) { std::fclose(filehandle); }
        };
        
    }
    
    /// RAII-ified simplifications, for opening directory and file handles
    /// ... also known by their obnoxiously-capitalized type names
    /// dating from the dark days of early C without the plus-plus--
    /// the original opaques: FILE* and DIR*,
    /// herein wrapped neatly out of sight forever in unique_ptrs.
    /// ... YOURE WELCOME.
    using directory = std::unique_ptr<typename std::decay<DIR>::type, detail::dircloser<DIR>>;
    using file = std::unique_ptr<typename std::decay<FILE>::type, detail::filecloser<FILE>>;
    
    namespace detail {
        /// We can construct the above unique_ptr type aliases directly from FILE* and DIR* --
        /// ... these are shortcuts that wrap calls to ::opendir() and ::fopen(), respectively;
        /// so you can be like:
        ///
        ///     filesystem::directory dir = detail::ddopen("the/path/to/it/");
        ///     /// dir will auto-close on scope exit
        ///     ::some_posix_func_that_wants_a_dirhandle(dir.get());
        ///     
        /// ... see? see what I am getting at with all this? NO DIR!! haha. anyway.
        filesystem::directory ddopen(const char *c);
        filesystem::directory ddopen(const std::string &s);
        filesystem::directory ddopen(const path &p);
        filesystem::file ffopen(const std::string &s, mode m = mode::READ);
    }

} /* namespace filesystem */

#endif /// LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_