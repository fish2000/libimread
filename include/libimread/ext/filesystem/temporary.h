/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_
#define LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_

#include <string>
#include <cstring>
#include <fstream>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    struct TemporaryName final {
        
        static constexpr char tfp[] = FILESYSTEM_TEMP_FILENAME;
        static constexpr char tfs[] = FILESYSTEM_TEMP_SUFFIX;
        
        int  descriptor;
        bool cleanup;
        bool deallocate;
        char* suffix;
        char* prefix;
        path  pathname;
        char* filename;
        
        explicit TemporaryName(char const* s = tfs, bool c = true,
                               char const* p = tfp,
                               path const& td = path::tmp());
        
        explicit TemporaryName(std::string const& s, bool c = true,
                               std::string const& p = tfp,
                               path const& td = path::tmp());
        
        TemporaryName(TemporaryName const& other);
        TemporaryName(TemporaryName&& other) noexcept;
        
        std::string   str() const noexcept;
        char const* c_str() const noexcept;
        operator std::string() const noexcept;
        operator char const*() const noexcept;
        
        bool create();
        bool exists();
        bool remove();
        char const* do_not_destroy();
        ~TemporaryName();
        
    };
    
    struct NamedTemporaryFile final {
        /// As per the eponymous tempfile.NamedTemporaryFile,
        /// of the Python standard library. NOW WITH RAII!!
        using openmode = std::ios_base::openmode;
        
        static constexpr char tfp[] = FILESYSTEM_TEMP_FILENAME;
        static constexpr char tfs[] = FILESYSTEM_TEMP_SUFFIX;
        
        int descriptor;
        bool cleanup;
        bool deallocate;
        char* suffix;
        char* prefix;
        mode filemode;
        path filepath;
        std::fstream stream;
        
        explicit NamedTemporaryFile(char const* s = tfs, bool c = true,
                                    char const* p = tfp,
                                    mode m = mode::WRITE,
                                    path const& td = path::tmp());
        
        explicit NamedTemporaryFile(std::string const& s, bool c = true,
                                    std::string const& p = tfp,
                                    mode m = mode::WRITE,
                                    path const& td = path::tmp());
        
        explicit NamedTemporaryFile(TemporaryName&& name);
        NamedTemporaryFile(NamedTemporaryFile const& other);
        NamedTemporaryFile(NamedTemporaryFile&& other) noexcept;
        
        std::string   str() const noexcept;
        char const* c_str() const noexcept;
        operator std::string() const noexcept;
        operator char const*() const noexcept;
        
        openmode mode() const noexcept;
        openmode mode(openmode additionally) const noexcept;
        bool open(openmode additionally = std::ios::trunc);
        bool reopen(openmode additionally = std::ios::in);
        bool close();
        bool create();
        bool exists();
        bool remove();
        char const* do_not_destroy();
        ~NamedTemporaryFile();
    
    };
    
    struct TemporaryDirectory final {
        
        static constexpr char tdp[] = FILESYSTEM_TEMP_DIRECTORYNAME;
        static constexpr char tfp[] = FILESYSTEM_TEMP_FILENAME;
        static constexpr char tfs[] = FILESYSTEM_TEMP_SUFFIX;
        
        char* tpl;
        bool cleanup;
        bool deallocate;
        path tplpath;
        path dirpath;
        
        explicit TemporaryDirectory(char const* t = tdp, bool c = true);
        explicit TemporaryDirectory(std::string const& t, bool c = true);
        TemporaryDirectory(TemporaryDirectory const& other);
        TemporaryDirectory(TemporaryDirectory&& other) noexcept;
        
        std::string   str() const noexcept;
        char const* c_str() const noexcept;
        operator std::string() const noexcept;
        operator char const*() const noexcept;
        
        NamedTemporaryFile get(std::string const& suffix = tfs,
                               std::string const& prefix = tfp,
                               mode m = mode::WRITE);
        
        bool create();
        bool clean();
        bool exists();
        bool remove();
        ~TemporaryDirectory();
    
    };

} /* namespace filesystem */


#endif /// LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_