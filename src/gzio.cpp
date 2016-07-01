/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
// #include <sys/mman.h>
#include <sys/stat.h>
#include <zlib.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/gzio.hh>

namespace im {
    
    constexpr int gzio_source_sink::READ_FLAGS;
    constexpr int gzio_source_sink::WRITE_FLAGS;
    constexpr int gzio_source_sink::WRITE_CREATE_MASK;
    
    int gzio_source_sink::open_read(char* p) const {
        return ::open(p, READ_FLAGS);
    }
    
    int gzio_source_sink::open_write(char* p, int mask) const {
        return ::open(p, WRITE_FLAGS, mask);
    }
    
    gzio_source_sink::gzio_source_sink() {}
    
    gzio_source_sink::gzio_source_sink(detail::gzhandle_t gz)
        :gzhandle(gz)
        {}
    
    gzio_source_sink::gzio_source_sink(int fd)
        :descriptor(fd)
        ,gzhandle(::gzdopen(descriptor, "FUCK MODES"))
        ,external(true)
        {}
    
    gzio_source_sink::~gzio_source_sink() { close(); }
    
    bool gzio_source_sink::can_seek() const noexcept { return true; } /// -ish (see SEEK_END note below)
    
    std::size_t gzio_source_sink::seek_absolute(std::size_t pos) { return ::gzseek(gzhandle, pos, SEEK_SET); }
    std::size_t gzio_source_sink::seek_relative(int delta) { return ::gzseek(gzhandle, delta, SEEK_CUR); }
    
    /// SEEK_END is, according to the zlib manual, UNSUPPORTED:
    std::size_t gzio_source_sink::seek_end(int delta) { return ::gzseek(gzhandle, delta, SEEK_END); }
    
    std::size_t gzio_source_sink::read(byte* buffer, std::size_t n) {
        int out = ::gzread(gzhandle, buffer, n);
        if (out == -1) {
            imread_raise(CannotReadError,
                "::gzread() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t gzio_source_sink::write(const void* buffer, std::size_t n) {
        int out = ::gzwrite(gzhandle, buffer, n);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "::write() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t gzio_source_sink::write(std::vector<byte> const& bv) {
        return this->write(
            static_cast<const void*>(&bv[0]),
            bv.size());
    }
    
    detail::stat_t gzio_source_sink::stat() const {
        if (!external) {
            imread_raise(CannotReadError,
                "cannot stat() on an internal gzhandler_t");
        }
            detail::stat_t info;
        if (::fstat(descriptor, &info) == -1) {
            imread_raise(CannotReadError,
                "::fstat() returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    void gzio_source_sink::flush() { ::gzflush(gzhandle, Z_SYNC_FLUSH); }
    
    std::size_t gzio_source_sink::size() const {
        detail::stat_t info = this->stat();
        return info.st_size * sizeof(byte);
    }
    
    void* gzio_source_sink::readmap(std::size_t pageoffset) const {
        /// HOW IS COMPRES MMAP FORMED
        imread_raise_default(NotImplementedError);
    }
    
    int gzio_source_sink::fd() const noexcept {
        return descriptor;
    }
    
    void gzio_source_sink::fd(int fd) noexcept {
        descriptor = fd;
        external = true;
    }
    
    bool gzio_source_sink::exists() const noexcept {
        if (!external) { return false; }
        try {
            this->stat();
        } catch (CannotReadError const& e) {
            return false;
        }
        return true;
    }
    
    int gzio_source_sink::open(char* cpath, filesystem::mode fmode) {
        if (fmode == filesystem::mode::WRITE) {
            descriptor = open_write(cpath);
            if (descriptor < 0) {
                imread_raise(CannotWriteError, "descriptor open-to-write failure:",
                    FF("\t::open(\"%s\", O_WRONLY | O_FSYNC | O_CREAT | O_EXCL | O_TRUNC)", cpath),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        } else {
            descriptor = open_read(cpath);
            if (descriptor < 0) {
                imread_raise(CannotReadError, "descriptor open-to-read failure:",
                    FF("\t::open(\"%s\", O_RDONLY | O_FSYNC)", cpath),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        }
        
        /// the 'T' in the 'wT' mode string stands for 'transparent writes',
        /// according to the zlib manual (which is for append-only type situations)
        char const* modestring = fmode == filesystem::mode::READ ? "rb" : "wb";
        gzhandle = ::gzdopen(descriptor, modestring);
        
        if (!gzhandle) {
            imread_raise(CannotReadError, "zlib stream-open failure:",
                FF("\t::gzopen(\"%s\", %s)", cpath, modestring),
                FF("\treturned negative value: %i", descriptor),
                    "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        external = true;
        return descriptor;
    }
    
    int gzio_source_sink::close() {
        using std::swap;
        int out = -1;
        if (::gzclose(gzhandle) != Z_OK) {
            if (descriptor > 0) { ::close(descriptor); }
            imread_raise(FileSystemError,
                "error closing gzhandle:",
                std::strerror(errno));
        } else if (descriptor > 0) {
            if (::close(descriptor) == -1) {
                imread_raise(FileSystemError,
                    "error closing file descriptor:",
                    std::strerror(errno));
            }
            swap(out, descriptor);
        }
        return out;
    }
    
    gzfile_source_sink::gzfile_source_sink(filesystem::mode fmode)
        :gzio_source_sink(), md(fmode)
        {}
    
    gzfile_source_sink::gzfile_source_sink(char* cpath, filesystem::mode fmode)
        :gzio_source_sink(), pth(cpath), md(fmode)
        {
            gzio_source_sink::open(cpath, fmode);
        }
    
    gzfile_source_sink::gzfile_source_sink(char const* ccpath, filesystem::mode fmode)
        :gzfile_source_sink(const_cast<char*>(ccpath), fmode)
        {}
    
    gzfile_source_sink::gzfile_source_sink(std::string& spath, filesystem::mode fmode)
        :gzfile_source_sink(spath.c_str(), fmode)
        {}
    
    gzfile_source_sink::gzfile_source_sink(std::string const& cspath, filesystem::mode fmode)
        :gzfile_source_sink(cspath.c_str(), fmode)
        {}
    
    gzfile_source_sink::gzfile_source_sink(filesystem::path const& ppath, filesystem::mode fmode)
        :gzfile_source_sink(ppath.c_str(), fmode)
        {}
    
    gzfile_source_sink::~gzfile_source_sink() {
        gzio_source_sink::close();
    }
    
    filesystem::path const& gzfile_source_sink::path() const {
        return pth;
    }
    
    bool gzfile_source_sink::exists() const noexcept {
        return pth.exists();
    }
    
    filesystem::mode gzfile_source_sink::mode(filesystem::mode m) {
        md = m;
        return md;
    }
    
    filesystem::mode gzfile_source_sink::mode() const {
        return md;
    }
    
    gzio::source::source()
        :gzfile_source_sink()
        {}
    
    gzio::source::source(char* cpath)
        :gzfile_source_sink(cpath)
        {}
    
    gzio::source::source(char const* ccpath)
        :gzfile_source_sink(ccpath)
        {}
    
    gzio::source::source(std::string& spath)
        :gzfile_source_sink(spath)
        {}
    
    gzio::source::source(std::string const& cspath)
        :gzfile_source_sink(cspath)
        {}
    
    gzio::source::source(filesystem::path const& ppath)
        :gzfile_source_sink(ppath)
        {}
    
    gzio::sink::sink()
        :gzfile_source_sink(filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(char* cpath)
        :gzfile_source_sink(cpath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(char const* ccpath)
        :gzfile_source_sink(ccpath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(std::string& spath)
        :gzfile_source_sink(spath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(std::string const& cspath)
        :gzfile_source_sink(cspath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(filesystem::path const& ppath)
        :gzfile_source_sink(ppath, filesystem::mode::WRITE)
        {}
    
}
