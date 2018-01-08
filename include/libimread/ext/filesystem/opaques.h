/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_
#define LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_

#include <string>
#include <memory>
#include <type_traits>

#include <cctype>
#include <cstdio>
#include <cstddef>
#include <dirent.h>

#include <libimread/ext/filesystem/mode.h>

namespace filesystem {
    
    class path; /// forward declaration for the following prototypes
    
    namespace detail {
        
        /// Deleter structures to close directory and file handles:
        
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
        
    } /// namespace detail
    
    /// RAII-ified simplifications, for opening directory and file handles
    /// ... also known by their obnoxiously-capitalized type names
    /// dating from the dark days of early C without the plus-plus--
    /// the original opaques: FILE* and DIR*,
    /// herein wrapped neatly out of sight forever in std::unique_ptrs.
    /// ... YOURE WELCOME.
    
    using directory  = detail::handle_ptr<DIR>;
    using file       = detail::handle_ptr<FILE>;
    
    namespace detail {
        
        using dirent_t = struct dirent;
        
        /// We can construct the above std::unique_ptr type aliases directly for FILE* and DIR* --
        /// ... these are shortcuts that wrap calls to ::opendir(), ::fdopendir(), std::fopen(),
        /// and ::fdopen(), respectively; so you can be like:
        ///     
        ///     {
        ///         filesystem::directory dir = detail::ddopen("the/path/to/it/");
        ///         /// dir will auto-close on scope exit
        ///         ::some_posix_func_that_wants_a_dirhandle(dir.get());
        ///     }
        ///     
        /// ... see? see what I am getting at with all this? NO DIR!! haha. anyway.
        
        filesystem::directory   ddopen(filesystem::path const& p);
        filesystem::directory   ddopen(std::string const& s);
        filesystem::directory   ddopen(int const descriptor);
        
        filesystem::file        ffopen(filesystem::path const& p,   mode m = mode::READ);
        filesystem::file        ffopen(std::string const& s,        mode m = mode::READ);
        filesystem::file        ffopen(int const descriptor,        mode m = mode::READ);
        
    } /// namespace detail
    
} /// namespace filesystem

#endif /// LIBIMREAD_EXT_FILESYSTEM_OPAQUES_H_