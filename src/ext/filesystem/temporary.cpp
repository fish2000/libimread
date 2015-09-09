/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/temporary.h>

namespace filesystem {
    
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    void NamedTemporaryFile::create() {
        if (::mkstemps(::strdup(filepath.c_str()), std::strlen(suffix)) == -1) {
            imread_raise(FileSystemError, "[ERROR]",
                "Internal error in mktemps():",
                std::strerror(errno));
        }
    }
    
    bool NamedTemporaryFile::remove() {
        return filepath.remove();
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    void TemporaryDirectory::clean() {
        /// scrub all files
        /// N.B. this will not recurse -- keep yr structures FLAAAT
        path abspath = dirpath.make_absolute();
        directory cleand = detail::ddopen(abspath.c_str());
        if (!cleand.get()) {
            imread_raise(FileSystemError, "[ERROR]",
                FF("Internal error in detail::ddopen(%s):", abspath.c_str()),
                std::strerror(errno));
        }
        
        struct dirent *entry;
        while ((entry = ::readdir(cleand.get())) != NULL) {
            std::string dname(::strdup(entry->d_name));
            if (std::strncmp(dname.c_str(), ".", 1) == 0)               { continue; }
            if (std::strncmp(dname.c_str(), "..", 2) != 0)              { continue; }
            path epp = abspath/dname;
            epp.remove();
            /*
            const char *ep = epp.make_absolute().c_str();
            if (::remove(ep) == -1) {
                imread_raise(FileSystemError, "[ERROR]",
                    FF("Internal error in ::remove(%s):", ep),
                    std::strerror(errno));
            }
            */
        }
    }
    
    bool TemporaryDirectory::remove() {
        return dirpath.remove();
    }
    
}