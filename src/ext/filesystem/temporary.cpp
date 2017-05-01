/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <unistd.h>

#include <vector>
#include <algorithm>

#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/pystring.hh>

namespace filesystem {
    
    DECLARE_CONSTEXPR_CHAR(TemporaryName::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryName::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    bool TemporaryName::create() {
        /// Create a new temporary file and return a descriptor
        descriptor = ::mkstemps(const_cast<char*>(pathname.c_str()),
                                std::strlen(suffix));
        
        /// Return false if temporary file creation went bad
        if (descriptor < 0) { return false; }
        
        /// Stash the normalized path name of the temporary file
        pathname = path(descriptor).make_absolute();
        filename = ::strdup(pathname.basename().c_str());
        
        /// Attempt to close the descriptor
        if (::close(descriptor) == -1) { return false; }
        
        /// Attempt to remove the file itself and return per the status
        return pathname.remove();
    }
    
    bool TemporaryName::exists() { return pathname.is_file(); }
    bool TemporaryName::remove() { return pathname.remove(); }
    
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
        return this->close() && this->open(additionally);
    }
    
    bool NamedTemporaryFile::close() {
        if (!stream.is_open()) { return false; }
        stream.close();
        return true;
    }
    
    bool NamedTemporaryFile::create() {
        descriptor = ::mkstemps(const_cast<char*>(filepath.c_str()), std::strlen(suffix));
        if (descriptor < 0) {
            return false;
        }
        filepath = path(descriptor);
        if (::close(descriptor) == -1) {
            return false;
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
            tplpath = path::join(path::tmp(), tpl).append("-XXXXXX");
        } else {
            tplpath = path::join(path::tmp(), tpl);
        }
        char const* dtemp = ::mkdtemp(const_cast<char*>(tplpath.c_str()));
        if (dtemp == nullptr) {
            return false;
        }
        dirpath = path(dtemp);
        return true;
    }
    
    bool TemporaryDirectory::clean() {
        if (!dirpath.exists()) { return false; }
        bool out = true;
        detail::pathvec_t dirs;
        
        /// perform walk with visitor --
        /// recursively removing files while saving directories
        /// as full paths in the `dirs` vector
        dirpath.make_absolute().walk([&out, &dirs](path const& p,
                                                   detail::stringvec_t& directories,
                                                   detail::stringvec_t& files) {
            if (!directories.empty()) {
                std::for_each(directories.begin(),
                              directories.end(),
                  [&p, &dirs](std::string const& d) {
                      dirs.emplace_back(p/d);
                });
            }
            if (!files.empty()) {
                std::for_each(files.begin(),
                              files.end(),
                   [&p, &out](std::string const& f) {
                      out &= (p/f).remove();
                });
            }
        });
        
        /// remove emptied directories per saved list
        if (!dirs.empty()) {
            /// reverse directorylist --
            std::reverse(dirs.begin(), dirs.end());
            /// -- removing uppermost directories top-down:
            std::for_each(dirs.begin(), dirs.end(),
                   [&out](path const& p) {
                out &= p.remove();
            });
        }
        
        /// return as per logical sum of `remove()` call successes
        return out;
    }
    
    bool TemporaryDirectory::remove() {
        return dirpath.remove();
    }
    
}
