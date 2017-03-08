/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_
#define LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_

#include <string>
#include <fstream>
#include <cstring>
#include <cstdlib>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
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
        
        explicit NamedTemporaryFile(char const* s = tfs, bool c = true, char const* p = tfp,
                                    mode m = mode::WRITE, path const& td = path::tmp())
                                        :cleanup(c), deallocate(true)
                                        ,suffix(::strdup(s)), prefix(::strdup(p))
                                        ,filemode(m)
                                        ,filepath(td/std::strcat(prefix, s))
                                        ,stream()
                                        {
                                            create();
                                        }
        
        explicit NamedTemporaryFile(std::string const& s, bool c = true, std::string const& p = tfp,
                                    mode m = mode::WRITE, path const& td = path::tmp())
                                        :cleanup(c), deallocate(true)
                                        ,suffix(::strdup(s.c_str())), prefix(::strdup(p.c_str()))
                                        ,filemode(m)
                                        ,filepath(td/(p+s))
                                        ,stream()
                                        {
                                            create();
                                        }
        
        NamedTemporaryFile(NamedTemporaryFile const& other)
            :descriptor(other.descriptor)
            ,cleanup(other.cleanup), deallocate(true)
            ,suffix(::strdup(other.suffix)), prefix(::strdup(other.prefix))
            ,filemode(other.filemode)
            ,filepath(other.filepath)
            ,stream()
            {}
        
        NamedTemporaryFile(NamedTemporaryFile&& other) noexcept
            :descriptor(other.descriptor)
            ,cleanup(other.cleanup), deallocate(true)
            ,suffix(std::move(other.suffix)), prefix(std::move(other.prefix))
            ,filemode(other.filemode)
            ,filepath(std::move(other.filepath))
            ,stream(std::move(other.stream))
            {
                other.cleanup = false;
                other.deallocate = false;
            }
        
        inline std::string   str() const noexcept  { return filepath.str(); }
        inline char const* c_str() const noexcept  { return filepath.c_str(); }
        operator std::string() const noexcept      { return str(); }
        operator char const*() const noexcept      { return c_str(); }
        
        inline openmode mode() const noexcept {
            return filemode == mode::WRITE ? std::ios::out : std::ios::in;
        }
        inline openmode mode(openmode additionally) const noexcept {
            return this->mode() | additionally;
        }
        
        bool open(openmode additionally = std::ios::trunc);
        bool reopen(openmode additionally = std::ios::in);
        bool close();
        
        bool create();
        bool remove();
        
        ~NamedTemporaryFile() {
            if (cleanup) { close(); remove(); }
            if (deallocate) { std::free(suffix);
                              std::free(prefix); }
        }
    
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
        
        explicit TemporaryDirectory(char const* t = tdp, bool c = true)
            :tpl(::strdup(t)), cleanup(c), deallocate(true)
            {
                create();
            }
        
        explicit TemporaryDirectory(std::string const& t, bool c = true)
            :tpl(::strdup(t.c_str())), cleanup(c), deallocate(true)
            {
                create();
            }
        
        TemporaryDirectory(TemporaryDirectory const& other)
            :tpl(::strdup(other.tpl))
            ,cleanup(other.cleanup), deallocate(true)
            ,tplpath(other.tplpath)
            ,dirpath(other.dirpath)
            {}
        
        TemporaryDirectory(TemporaryDirectory&& other) noexcept
            :tpl(std::move(other.tpl))
            ,cleanup(other.cleanup), deallocate(true)
            ,tplpath(std::move(other.tplpath))
            ,dirpath(std::move(other.dirpath))
            {
                other.cleanup = false;
                other.deallocate = false;
            }
        
        inline std::string   str() const noexcept   { return dirpath.str(); }
        inline char const* c_str() const noexcept   { return dirpath.c_str(); }
        operator std::string() const noexcept       { return str(); }
        operator char const*() const noexcept       { return c_str(); }
        
        NamedTemporaryFile get(std::string const& suffix = tfs,
                               std::string const& prefix = tfp,
                               mode m = mode::WRITE) { return NamedTemporaryFile(
                                                          suffix, cleanup,
                                                          prefix, m, dirpath); }
        
        bool create();
        bool clean();
        bool remove();
        
        ~TemporaryDirectory() {
            if (cleanup)    { clean(); remove(); }
            if (deallocate) { std::free(tpl); }
        }
    
    };

} /* namespace filesystem */


#endif /// LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_