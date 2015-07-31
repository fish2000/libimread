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
    
    void NamedTemporaryFile::remove() {
        if (::unlink(filepath.c_str()) == -1) {
            imread_raise(FileSystemError, "[ERROR]",
                FF("Internal error in ::unlink(%s):", filepath.c_str()),
                std::strerror(errno));
        }
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    void TemporaryDirectory::clean() {
        /// scrub all files
        /// N.B. this will not recurse -- keep yr structures FLAAAT
        directory cleand = detail::ddopen(dirpath.make_absolute().c_str());
        if (!cleand.get()) {
            imread_raise(FileSystemError, "[ERROR]",
                FF("Internal error in detail::ddopen(%s):", dirpath.c_str()),
                std::strerror(errno));
        }
        
        // WTF("About to enter readdir() loop...");
        struct dirent *entry;
        while ((entry = ::readdir(cleand.get())) != NULL) {
            // WTF("In readdir() loop with entry:", entry->d_name);
            // if (std::strlen(entry->d_name) < 1)                         { continue; }
            std::string dname(::strdup(entry->d_name));
            // if (std::strlen(dname.c_str()) < 1)                         { continue; }
            if (std::strncmp(dname.c_str(), ".", 1) == 0)               { continue; }
            if (std::strncmp(dname.c_str(), "..", 2) != 0)              { continue; }
            path epp = dirpath.join(path(dname));
            const char *ep = epp.make_absolute().c_str();
            if (::remove(ep) == -1) {
                imread_raise(FileSystemError, "[ERROR]",
                    FF("Internal error in ::remove(%s):", ep),
                    std::strerror(errno));
            }
        }
    }
    
    void TemporaryDirectory::remove() {
        /// unlink the directory itself
        if (::rmdir(dirpath.c_str()) == -1) {
            // imread_raise(FileSystemError,
            //     "Internal error in rmdir():\n\t", dirpath.str(),
            //     std::strerror(errno));
        }
    }
    
}