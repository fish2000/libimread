/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_
#define LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_

#include <string>
#include <cstring>
#include <cstdlib>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    struct NamedTemporaryFile {
        /// As per the eponymous tempfile.NamedTemporaryFile,
        /// of the Python standard library. NOW WITH RAII!!
        
        static constexpr char tfp[] = FILESYSTEM_TEMP_FILENAME;
        static constexpr char tfs[] = FILESYSTEM_TEMP_SUFFIX;
        
        mode mm;
        char *suffix;
        char *prefix;
        bool cleanup;
        path filepath;
        int descriptor;
        
        explicit NamedTemporaryFile(const char *s = tfs, const char *p = tfp,
                                    const path &td = path::tmp(), bool c = true, mode m = mode::WRITE)
                                        :mm(m), cleanup(c), suffix(::strdup(s)), prefix(::strdup(p))
                                        ,filepath(td/std::strcat(prefix, s))
                                        {
                                            create();
                                        }
        explicit NamedTemporaryFile(const std::string &s, const std::string &p = tfp,
                                    const path &td = path::tmp(), bool c = true, mode m = mode::WRITE)
                                        :mm(m), cleanup(c), suffix(::strdup(s.c_str())), prefix(::strdup(p.c_str()))
                                        ,filepath(td/(p+s))
                                        {
                                            create();
                                        }
        
        inline std::string   str() const noexcept  { return filepath.str(); }
        inline const char *c_str() const noexcept  { return filepath.c_str(); }
        operator std::string() const noexcept      { return str(); }
        operator const char*() const noexcept      { return c_str(); }
        
        bool create();
        bool remove();
        
        ~NamedTemporaryFile() {
            if (cleanup) { remove(); }
            free(suffix);
            free(prefix);
        }
    
    };
    
    struct TemporaryDirectory {
        
        static constexpr char tdp[] = FILESYSTEM_TEMP_DIRECTORYNAME;
        static constexpr char tfp[] = FILESYSTEM_TEMP_FILENAME;
        static constexpr char tfs[] = FILESYSTEM_TEMP_SUFFIX;
        
        char *tpl;
        bool cleanup;
        path tplpath;
        path dirpath;
        
        explicit TemporaryDirectory(const char *t, bool c = true)
            :tpl(::strdup(t))
            ,cleanup(c)
            ,tplpath(path::join(path::gettmp(), path(t)))
            {
                create();
            }
        
        explicit TemporaryDirectory(const std::string &t, bool c = true)
            :tpl(::strdup(t.c_str()))
            ,cleanup(c)
            ,tplpath(path::join(path::gettmp(), path(t)))
            {
                create();
            }
        
        inline std::string   str() const noexcept   { return dirpath.str(); }
        inline const char *c_str() const noexcept   { return dirpath.c_str(); }
        operator std::string() const noexcept       { return str(); }
        operator const char*() const noexcept       { return c_str(); }
        
        NamedTemporaryFile get(const std::string &suffix = tfs,
                               const std::string &prefix = tfp,
                               mode m = mode::WRITE) { return NamedTemporaryFile(
                                                          suffix, prefix, dirpath, cleanup, m); }
        
        bool create();
        bool clean();
        bool remove();
        
        ~TemporaryDirectory() {
            if (cleanup) { clean(); remove(); }
            free(tpl);
        }
    
    };

} /* namespace filesystem */


#endif /// LIBIMREAD_EXT_FILESYSTEM_TEMPORARY_H_