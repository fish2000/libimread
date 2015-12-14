/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cerrno>
#include <cstddef>
#include <dirent.h>
#include <fcntl.h>

#include <vector>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/pystring.hh>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/filesystem/opaques.h>

namespace filesystem {
    
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    bool NamedTemporaryFile::create() {
        descriptor = ::mkstemps(const_cast<char*>(filepath.c_str()), std::strlen(suffix));
        if (descriptor == -1) { return false; }
        filepath = path(descriptor);
        if (::close(descriptor) == -1) {
            imread_raise(FileSystemError,
                "NamedTemporaryFile::create(): error while closing descriptor:",
                std::strerror(errno));
        }
        return true;
    }
    
    bool NamedTemporaryFile::remove() {
        return filepath.remove();
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    bool TemporaryDirectory::create() {
        if (!pystring::endswith(tpl, "XXX")) {
            tplpath = path::join(path::gettmp(), path(tpl)).append("-XXXXXX");
        } else {
            tplpath = path::join(path::gettmp(), path(tpl));
        }
        const char* dtemp = ::mkdtemp(const_cast<char*>(tplpath.c_str()));
        if (dtemp == NULL) { return false; }
        dirpath = path(dtemp);
        return true;
    }
    
    bool TemporaryDirectory::cleand() {
        /// scrub all files
        /// N.B. this will not recurse -- keep yr structures FLAAAT
        if (!dirpath.exists()) { return false; }
        path abspath = dirpath.make_absolute();
        directory cleand = detail::ddopen(abspath.c_str());
        bool out = true;
        
        if (!cleand.get()) {
            imread_raise(FileSystemError, "[ERROR]",
                "TemporaryDirectory::clean(): error in detail::ddopen() with path:",
                abspath.str(), std::strerror(errno));
        }
        
        detail::dirent_t* entry;
        while ((entry = ::readdir(cleand.get())) != NULL) {
            std::string dname(entry->d_name);
            if (std::strncmp(dname.c_str(), ".", 1) == 0)   { continue; }
            if (std::strncmp(dname.c_str(), "..", 2) == 0)  { continue; }
            path epp = abspath/dname;
            out &= epp.remove();
        }
        
        return out;
    }
    
    bool TemporaryDirectory::clean() {
        if (!dirpath.exists()) { return false; }
        bool out = true;
        detail::pathvec_t directorylist;
        path abspath = dirpath.make_absolute();
        
        /// walk_visitor_t recursively removes files while saving directories
        /// as full paths in the `directorylist` vector
        detail::walk_visitor_t walk_visitor = [&out, &directorylist](
                                    const path& p,
                                    detail::stringvec_t& directories,
                                    detail::stringvec_t& files) {
            if (!directories.empty()) {
                std::for_each(directories.begin(), directories.end(),
                    [&p, &directorylist](const std::string& directory) {
                        directorylist.push_back(p/directory); });
            }
            if (!files.empty()) {
                std::for_each(files.begin(), files.end(),
                    [&p, &out](const std::string& file) {
                        out &= (p/file).remove(); });
            }
        };
        
        /// perform walk with visitor
        abspath.walk(std::forward<detail::walk_visitor_t>(walk_visitor));
        
        /// remove emptied directories per saved list
        if (!directorylist.empty()) {
            std::for_each(directorylist.begin(), directorylist.end(),
                   [&out](const path& p) { out &= p.remove(); });
        }
        
        /// return as per logical sum of `remove()` call successes
        return out;
    }
    
    bool TemporaryDirectory::remove() {
        return dirpath.remove();
    }
    
}