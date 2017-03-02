/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_
#define LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_

#include <cctype>
#include <cstdio>
#include <cstddef>
#include <dirent.h>

#include <string>
#include <memory>
#include <type_traits>

#include <libimread/ext/filesystem/mode.h>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        /// Deleter structures to close directory and file handles
        template <typename HandleType>
        struct handle_helper;
        
        template <>
        struct handle_helper<DIR>  {
            using ptr_t = std::add_pointer_t<DIR>;
            static void close(ptr_t handle) { ::closedir(handle); }
        };
        
        template <>
        struct handle_helper<FILE> {
            using ptr_t = std::add_pointer_t<FILE>;
            static void close(ptr_t handle) { std::fclose(handle); }
        };
        
        template <typename HandleType>
        using handle_t = typename handle_helper<HandleType>::ptr_t;
        
        template <typename T>
        struct closer {
            using opaque_t = std::decay_t<T>;
            constexpr closer() noexcept = default;
            template <typename U> closer(closer<U> const&) noexcept {};
            void operator()(handle_t<T> handle) { handle_helper<T>::close(handle); }
        };
        
        template <typename T>
        using handle_ptr = std::unique_ptr<typename closer<T>::opaque_t,
                                                    closer<T>>;
    }
    
    /// RAII-ified simplifications, for opening directory and file handles
    /// ... also known by their obnoxiously-capitalized type names
    /// dating from the dark days of early C without the plus-plus--
    /// the original opaques: FILE* and DIR*,
    /// herein wrapped neatly out of sight forever in unique_ptrs.
    /// ... YOURE WELCOME.
    using directory  = detail::handle_ptr<DIR>;
    using file       = detail::handle_ptr<FILE>;
    
    namespace detail {
        using dirent_t = struct dirent;
        
        /// We can construct the above unique_ptr type aliases directly from FILE* and DIR* --
        /// ... these are shortcuts that wrap calls to ::opendir() and std::fopen(), respectively;
        /// so you can be like:
        ///
        ///     filesystem::directory dir = detail::ddopen("the/path/to/it/");
        ///     /// dir will auto-close on scope exit
        ///     ::some_posix_func_that_wants_a_dirhandle(dir.get());
        ///     
        /// ... see? see what I am getting at with all this? NO DIR!! haha. anyway.
        directory ddopen(char const* c);
        directory ddopen(std::string const& s);
        directory ddopen(path const& p);
        filesystem::file ffopen(std::string const& s, mode m = mode::READ);
        
        static constexpr int EXECUTION_BUFFER_SIZE = 128;
        std::string execute(char const* command);
    }

} /* namespace filesystem */

#endif /// LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_