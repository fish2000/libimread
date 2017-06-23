/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstdio>
#include <unistd.h>

#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/pystring.hh>

namespace filesystem {
    
    DECLARE_CONSTEXPR_CHAR(TemporaryName::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryName::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    TemporaryName::TemporaryName(char const* s, bool c,
                                 char const* p, path const& td)
                                    :descriptor(-1), cleanup(c), deallocate(true)
                                    ,suffix(::strdup(s)), prefix(::strdup(p))
                                    ,pathname(td/std::strcat(prefix, s))
                                    {
                                        create();
                                    }
    
    TemporaryName::TemporaryName(std::string const& s, bool c,
                                 std::string const& p, path const& td)
                                    :descriptor(-1), cleanup(c), deallocate(true)
                                    ,suffix(::strdup(s.c_str())), prefix(::strdup(p.c_str()))
                                    ,pathname(td/(p+s))
                                    {
                                        create();
                                    }
    
    TemporaryName::TemporaryName(TemporaryName const& other)
        :descriptor(other.descriptor)
        ,cleanup(other.cleanup), deallocate(true)
        ,suffix(::strdup(other.suffix)), prefix(::strdup(other.prefix))
        ,pathname(other.pathname)
        ,filename(::strdup(other.filename))
        {}
    
    TemporaryName::TemporaryName(TemporaryName&& other) noexcept
        :descriptor(other.descriptor)
        ,cleanup(other.cleanup), deallocate(true)
        ,suffix(std::move(other.suffix)), prefix(std::move(other.prefix))
        ,pathname(std::move(other.pathname))
        ,filename(std::move(other.filename))
        {
            other.cleanup = false;
            other.deallocate = false;
        }
    
    std::string   TemporaryName::str() const noexcept       { return pathname.str(); }
    char const* TemporaryName::c_str() const noexcept       { return pathname.c_str(); }
    TemporaryName::operator std::string() const noexcept    { return str(); }
    TemporaryName::operator char const*() const noexcept    { return c_str(); }
    
    bool TemporaryName::create() {
        /// Create a new temporary file, storing the descriptor
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
    
    bool TemporaryName::exists() { return pathname.exists(); }
    bool TemporaryName::remove() { return pathname.rm_rf(); }
    
    char const* TemporaryName::do_not_destroy() {
        cleanup = false;
        return filename;
    }
    
    TemporaryName::~TemporaryName() {
        if (cleanup && exists()) { remove(); }
        if (deallocate) { std::free(suffix);
                          std::free(prefix);
                          std::free(filename); }
    }
    
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(NamedTemporaryFile::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    NamedTemporaryFile::NamedTemporaryFile(char const* s, bool c,
                                           char const* p, enum mode m,
                                           path const& td)
                                            :descriptor(-1), cleanup(c), deallocate(true)
                                            ,suffix(::strdup(s)), prefix(::strdup(p))
                                            ,filemode(m)
                                            ,filepath(td/std::strcat(prefix, s))
                                            ,stream()
                                            {
                                                create();
                                            }
    
    NamedTemporaryFile::NamedTemporaryFile(std::string const& s, bool c,
                                           std::string const& p, enum mode m,
                                           path const& td)
                                            :descriptor(-1), cleanup(c), deallocate(true)
                                            ,suffix(::strdup(s.c_str())), prefix(::strdup(p.c_str()))
                                            ,filemode(m)
                                            ,filepath(td/(p+s))
                                            ,stream()
                                            {
                                                create();
                                            }
    
    NamedTemporaryFile::NamedTemporaryFile(TemporaryName&& name)
        :descriptor(name.descriptor)
        ,cleanup(name.cleanup), deallocate(true)
        ,suffix(::strdup(name.suffix)), prefix(::strdup(name.prefix))
        ,filemode(mode::WRITE)
        ,filepath(name.pathname)
        ,stream()
        {
            create();
        }
    
    NamedTemporaryFile::NamedTemporaryFile(NamedTemporaryFile const& other)
        :descriptor(other.descriptor)
        ,cleanup(other.cleanup), deallocate(true)
        ,suffix(::strdup(other.suffix)), prefix(::strdup(other.prefix))
        ,filemode(other.filemode)
        ,filepath(other.filepath)
        ,stream()
        {}
    
    NamedTemporaryFile::NamedTemporaryFile(NamedTemporaryFile&& other) noexcept
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
        /// Create a new temporary file, storing the descriptor
        descriptor = ::mkstemps(const_cast<char*>(filepath.c_str()),
                                std::strlen(suffix));
        
        /// Return false if temporary file creation went bad
        if (descriptor < 0) { return false; }
        
        /// Stash the normalized path of the temporary file
        filepath = path(descriptor).make_absolute();
        
        /// Attempt to close the descriptor
        if (::close(descriptor) == -1) { return false; }
        
        /// All is well, return as such
        return true;
    }
    
    bool NamedTemporaryFile::exists() { return filepath.is_file(); }
    bool NamedTemporaryFile::remove() { return filepath.remove(); }
    
    char const* NamedTemporaryFile::do_not_destroy() {
        cleanup = false;
        return filepath.basename().c_str();
    }
    
    NamedTemporaryFile::~NamedTemporaryFile() {
        if (cleanup) { close(); remove(); }
        if (deallocate) { std::free(suffix);
                          std::free(prefix); }
    }
    
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tdp, FILESYSTEM_TEMP_DIRECTORYNAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfp, FILESYSTEM_TEMP_FILENAME);
    DECLARE_CONSTEXPR_CHAR(TemporaryDirectory::tfs, FILESYSTEM_TEMP_SUFFIX);
    
    TemporaryDirectory::TemporaryDirectory(char const* t, bool c)
        :tpl(::strdup(t)), cleanup(c), deallocate(true)
        {
            create();
        }
    
    TemporaryDirectory::TemporaryDirectory(std::string const& t, bool c)
        :tpl(::strdup(t.c_str())), cleanup(c), deallocate(true)
        {
            create();
        }
    
    TemporaryDirectory::TemporaryDirectory(TemporaryDirectory const& other)
        :tpl(::strdup(other.tpl))
        ,cleanup(other.cleanup), deallocate(true)
        ,tplpath(other.tplpath)
        ,dirpath(other.dirpath)
        {}
    
    TemporaryDirectory::TemporaryDirectory(TemporaryDirectory&& other) noexcept
        :tpl(std::move(other.tpl))
        ,cleanup(other.cleanup), deallocate(true)
        ,tplpath(std::move(other.tplpath))
        ,dirpath(std::move(other.dirpath))
        {
            other.cleanup = false;
            other.deallocate = false;
        }
    
    std::string   TemporaryDirectory::str() const noexcept      { return dirpath.str(); }
    char const* TemporaryDirectory::c_str() const noexcept      { return dirpath.c_str(); }
    TemporaryDirectory::operator std::string() const noexcept   { return str(); }
    TemporaryDirectory::operator char const*() const noexcept   { return c_str(); }
    
    NamedTemporaryFile TemporaryDirectory::get(std::string const& suffix,
                                               std::string const& prefix,
                                               enum mode m) { return NamedTemporaryFile(suffix,
                                                                                        cleanup,
                                                                                        prefix, m, dirpath); }
    
    bool TemporaryDirectory::create() {
        /// Create a new path in the temporary directory,
        /// suffixed properly for mkdtemp()
        if (!pystring::endswith(tpl, "XXX")) {
            tplpath = path::tmp().join(tpl).append("-XXXXXX");
        } else {
            tplpath = path::tmp().join(tpl);
        }
        
        /// Actually create the temporary directory
        char const* dtemp = ::mkdtemp(const_cast<char*>(tplpath.c_str()));
        
        /// Return false if the temporary directory creation went bad
        if (dtemp == nullptr) { return false; }
        
        /// Stash the normalized path of the temporary directory
        dirpath = path(dtemp).make_absolute();
        
        /// All is well, return as such
        return true;
    }
    
    bool TemporaryDirectory::clean() {
        if (!dirpath.exists()) { return false; }
        return dirpath.rm_rf();
    }
    
    bool TemporaryDirectory::exists() { return dirpath.is_directory(); }
    bool TemporaryDirectory::remove() { return dirpath.remove(); }
    
    TemporaryDirectory::~TemporaryDirectory() {
        if (cleanup)    { clean(); remove(); }
        if (deallocate) { std::free(tpl); }
    }
    
}
