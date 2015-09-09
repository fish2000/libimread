/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/temporary.h>

namespace filesystem {
    
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    bool NamedTemporaryFile::create() {
        return (::mkstemps(::strdup(filepath.c_str()), std::strlen(suffix)) != -1);
    }
    
    bool NamedTemporaryFile::remove() {
        return filepath.remove();
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    bool TemporaryDirectory::clean() {
        /// scrub all files
        /// N.B. this will not recurse -- keep yr structures FLAAAT
        if (!dirpath.exists()) { return false; }
        path abspath = dirpath.make_absolute();
        directory cleand = detail::ddopen(abspath.c_str());
        bool out = true;
        
        if (!cleand.get()) {
            imread_raise(FileSystemError, "[ERROR]",
                FF("TemporaryDirectory::clean(): error in detail::ddopen(%s):", abspath.c_str()),
                std::strerror(errno));
        }
        
        struct dirent *entry;
        while ((entry = ::readdir(cleand.get())) != NULL) {
            std::string dname(::strdup(entry->d_name));
            if (std::strncmp(dname.c_str(), ".", 1) == 0)   { continue; }
            if (std::strncmp(dname.c_str(), "..", 2) != 0)  { continue; }
            path epp = abspath/dname;
            out &= epp.remove();
        }
        
        return out;
    }
    
    bool TemporaryDirectory::remove() {
        return dirpath.remove();
    }
    
}