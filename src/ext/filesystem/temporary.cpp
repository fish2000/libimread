/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

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
    
    bool NamedTemporaryFile::open(openmode additionally) {
        stream.open(filepath.str(), this->mode(additionally));
        if (!stream.is_open()) {
            stream.clear();
            stream.open(filepath.str(), std::ios::out); /// TOUCH!
            stream.close();
            stream.open(filepath.str(), this->mode(additionally));
        }
        return stream.good();
    }
    
    bool NamedTemporaryFile::reopen(openmode additionally) {
        close();
        return open(additionally);
    }
    
    bool NamedTemporaryFile::close() {
        if (!stream.is_open()) { return false; }
        stream.close();
        return true;
    }
    
    bool NamedTemporaryFile::create() {
        descriptor = ::mkstemps(const_cast<char*>(filepath.c_str()), std::strlen(suffix));
        if (descriptor == -1) { return false; }
        filepath = filesystem::path(descriptor);
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
            tplpath = path::join(filesystem::path::gettmp(), filesystem::path(tpl)).append("-XXXXXX");
        } else {
            tplpath = path::join(filesystem::path::gettmp(), filesystem::path(tpl));
        }
        char const* dtemp = ::mkdtemp(const_cast<char*>(tplpath.c_str()));
        if (dtemp == nullptr) { return false; }
        dirpath = filesystem::path(dtemp);
        return true;
    }
    
    bool TemporaryDirectory::clean() {
        if (!dirpath.exists()) { return false; }
        bool out = true;
        detail::pathvec_t directorylist;
        filesystem::path abspath = dirpath.make_absolute();
        
        /// walk_visitor_t recursively removes files while saving directories
        /// as full paths in the `directorylist` vector
        detail::walk_visitor_t walk_visitor = [&out, &directorylist](
                                    filesystem::path const& p,
                                    detail::stringvec_t& directories,
                                    detail::stringvec_t& files) {
            if (!directories.empty()) {
                std::for_each(directories.begin(), directories.end(),
                    [&p, &directorylist](std::string const& directory) {
                        directorylist.push_back(p/directory); });
            }
            if (!files.empty()) {
                std::for_each(files.begin(), files.end(),
                    [&p, &out](std::string const& file) {
                        out &= (p/file).remove(); });
            }
        };
        
        /// perform walk with visitor
        abspath.walk(std::forward<detail::walk_visitor_t>(walk_visitor));
        
        /// remove emptied directories per saved list
        if (!directorylist.empty()) {
            /// reverse directorylist --
            std::reverse(directorylist.begin(),
                         directorylist.end());
            /// -- removing uppermost directories top-down:
            std::for_each(directorylist.begin(),
                          directorylist.end(),
                   [&out](filesystem::path const& p) { out &= p.remove(); });
        }
        
        /// return as per logical sum of `remove()` call successes
        return out;
    }
    
    bool TemporaryDirectory::remove() {
        return dirpath.remove();
    }
    
}