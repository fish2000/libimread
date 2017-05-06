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
    
    std::string   TemporaryName::str() const noexcept       { return pathname.str(); }
    char const* TemporaryName::c_str() const noexcept       { return pathname.c_str(); }
    TemporaryName::operator std::string() const noexcept    { return str(); }
    TemporaryName::operator char const*() const noexcept    { return c_str(); }
    
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
    bool TemporaryName::remove() { return pathname.rm_rf(); }
    
    char const* TemporaryName::do_not_destroy() {
        cleanup = false;
        return filename;
    }
    
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    std::string   NamedTemporaryFile::str() const noexcept      { return filepath.str(); }
    char const* NamedTemporaryFile::c_str() const noexcept      { return filepath.c_str(); }
    NamedTemporaryFile::operator std::string() const noexcept   { return str(); }
    NamedTemporaryFile::operator char const*() const noexcept   { return c_str(); }
    
    NamedTemporaryFile::openmode NamedTemporaryFile::mode() const noexcept {
        return filemode == mode::WRITE ? std::ios::out : std::ios::in;
    }
    
    NamedTemporaryFile::openmode NamedTemporaryFile::mode(NamedTemporaryFile::openmode additionally) const noexcept {
        return this->mode() | additionally;
    }
    
    bool NamedTemporaryFile::open(NamedTemporaryFile::openmode additionally) {
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
        descriptor = ::mkstemps(const_cast<char*>(filepath.c_str()),
                                std::strlen(suffix));
        
        if (descriptor < 0) { return false; }
        
        filepath = path(descriptor).make_absolute();
        
        if (::close(descriptor) == -1) { return false; }
        return true;
    }
    
    bool NamedTemporaryFile::exists() { return filepath.is_file(); }
    bool NamedTemporaryFile::remove() { return filepath.remove(); }
    
    char const* NamedTemporaryFile::do_not_destroy() {
        cleanup = false;
        return filepath.basename().c_str();
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    std::string   TemporaryDirectory::str() const noexcept      { return dirpath.str(); }
    char const* TemporaryDirectory::c_str() const noexcept      { return dirpath.c_str(); }
    TemporaryDirectory::operator std::string() const noexcept   { return str(); }
    TemporaryDirectory::operator char const*() const noexcept   { return c_str(); }
    
    bool TemporaryDirectory::create() {
        if (!pystring::endswith(tpl, "XXX")) {
            tplpath = path::tmp().join(tpl).append("-XXXXXX");
        } else {
            tplpath = path::tmp().join(tpl);
        }
        
        char const* dtemp = ::mkdtemp(const_cast<char*>(tplpath.c_str()));
        if (dtemp == nullptr) { return false; }
        
        dirpath = path(dtemp).make_absolute();
        return true;
    }
    
    bool TemporaryDirectory::clean() {
        if (!dirpath.exists()) { return false; }
        return dirpath.rm_rf();
    }
    
    bool TemporaryDirectory::exists() { return dirpath.is_directory(); }
    bool TemporaryDirectory::remove() { return dirpath.remove(); }
    
}
