/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/temporary.h>

namespace filesystem {
    
    constexpr char NamedTemporaryFile::tfp[im::static_strlen(FILESYSTEM_TEMP_FILENAME)];
    constexpr char NamedTemporaryFile::tfs[im::static_strlen(FILESYSTEM_TEMP_SUFFIX)];
    
    void NamedTemporaryFile::create() {
        int out = ::mkstemps(strdup(tf.c_str()), std::strlen(suffix));
        if (out == -1) {
            imread_raise(FileSystemError,
                "Internal error in mktemps():",
                std::strerror(errno));
        }
    }
    
    void NamedTemporaryFile::remove() {
        if (::unlink(tf.c_str()) == -1) {
            imread_raise(FileSystemError,
                "Internal error in unlink():",
                std::strerror(errno));
        }
    }
    
    constexpr char TemporaryDirectory::tdp[im::static_strlen(FILESYSTEM_TEMP_DIRECTORYNAME)];
    constexpr char TemporaryDirectory::tfp[im::static_strlen(FILESYSTEM_TEMP_FILENAME)];
    constexpr char TemporaryDirectory::tfs[im::static_strlen(FILESYSTEM_TEMP_SUFFIX)];
    
    void TemporaryDirectory::clean() {
        /// scrub all files
        /// N.B. this will not recurse -- keep yr structures FLAAAT
        directory cleand = detail::ddopen(td);
        if (!cleand.get()) {
            imread_raise(FileSystemError,
                "Internal error in opendir():",
                std::strerror(errno));
        }
        struct dirent *entry;
        while ((entry = ::readdir(cleand.get())) != NULL) {
            if (std::strncmp(entry->d_name, ".", 1) != 0 && std::strncmp(entry->d_name, "..", 2) != 0) {
                const char *ep = (td/entry->d_name).c_str();
                if (::access(ep, R_OK) != -1) {
                    if (::unlink(ep) == -1) {
                        imread_raise(FileSystemError,
                            "Internal error in unlink():",
                            std::strerror(errno));
                    }
                } else {
                    imread_raise(FileSystemError,
                        "Internal error in access():",
                        std::strerror(errno));
                }
            }
        }
    }
    
    void TemporaryDirectory::remove() {
        /// unlink the directory itself
        if (::rmdir(td.c_str()) == -1) {
            imread_raise(FileSystemError,
                "Internal error in rmdir():",
                std::strerror(errno));
        }
    }
    
}