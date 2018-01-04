/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/file.hh>
#include <libimread/ext/filesystem/attributes.h>

namespace im {
    
    namespace detail {
        using rlimit_t = struct rlimit;
    }
    
    constexpr int fd_source_sink::kReadFlags;
    constexpr int fd_source_sink::kWriteFlags;
    constexpr int fd_source_sink::kFIFOReadFlags;
    constexpr int fd_source_sink::kFIFOWriteFlags;
    constexpr int fd_source_sink::kWriteCreateMask;
    
    int fd_source_sink::open_read(char const* p) {
        return ::open(p, kReadFlags);
    }
    
    int fd_source_sink::open_write(char const* p, int mask) {
        return ::open(p, kWriteFlags, mask);
    }
    
    int fd_source_sink::fifo_open_read(char const* p) {
        return ::open(p, kFIFOReadFlags);
    }
    
    int fd_source_sink::fifo_open_write(char const* p) {
        return ::open(p, kFIFOWriteFlags);
    }
    
    bool fd_source_sink::check_descriptor(int fd) {
        return (::fcntl(fd, F_GETFD) != -1);
    }
    
    std::size_t fd_source_sink::max_descriptor_count() {
        detail::rlimit_t rl;
        if (::getrlimit(RLIMIT_NOFILE, &rl) == -1) {
            imread_raise(MetadataReadError,
                "descriptor limit value read failure",
                "\t::getrlimit(RLIMIT_NOFILE, &rl) returned -1",
                "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        return rl.rlim_cur;
    }
    
    std::size_t fd_source_sink::max_descriptor_count(std::size_t new_max) {
        detail::rlimit_t rl;
        rl.rlim_cur = new_max;
        rl.rlim_max = new_max;
        if (::setrlimit(RLIMIT_NOFILE, &rl) == -1) {
            imread_raise(MetadataWriteError,
                "descriptor limit value write failure",
             FF("\t::setrlimit(RLIMIT_NOFILE, &rl) [new_max = %u] returned -1", new_max),
                "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        return new_max;
    }
    
    fd_source_sink::fd_source_sink(int fd) noexcept
        :descriptor{ fd }
        {}
    
    fd_source_sink::fd_source_sink(fd_source_sink const& other)
        :descriptor{ ::dup(other.descriptor) }
        {}
    
    fd_source_sink::fd_source_sink(fd_source_sink&& other) noexcept
        :descriptor{ ::dup2(other.descriptor,
                            other.descriptor) }
            ,mapped{ std::exchange(other.mapped,
                                 detail::mapped_t{ nullptr }) }
        { other.descriptor = -1; }
    
    fd_source_sink::~fd_source_sink() { close(); }
    
    bool fd_source_sink::can_seek() const noexcept { return true; }
    bool fd_source_sink::can_store() const noexcept { return true; }
    
    std::size_t fd_source_sink::seek_absolute(std::size_t pos) { return ::lseek(descriptor, pos, SEEK_SET); }
    std::size_t fd_source_sink::seek_relative(int delta) { return ::lseek(descriptor, delta, SEEK_CUR); }
    std::size_t fd_source_sink::seek_end(int delta) { return ::lseek(descriptor, delta, SEEK_END); }
    
    std::size_t fd_source_sink::read(byte* buffer, std::size_t n) const {
        int out = ::read(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotReadError,
                "::read() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t fd_source_sink::write(const void* buffer, std::size_t n) {
        int out = ::write(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "::write() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t fd_source_sink::write(bytevec_t const& bv) {
        return this->write(
            static_cast<const void*>(&bv[0]),
            bv.size());
    }
    
    detail::stat_t fd_source_sink::stat() const {
        detail::stat_t info;
        if (::fstat(descriptor, &info) == -1) {
            imread_raise(CannotReadError,
                "::fstat() returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    void fd_source_sink::flush() { ::fsync(descriptor); }
    
    bytevec_t fd_source_sink::full_data() const {
        /// grab stat struct and store initial seek position
        detail::stat_t info = this->stat();
        std::size_t fsize = info.st_size * sizeof(byte);
        std::size_t orig = ::lseek(descriptor, 0, SEEK_CUR);
        
        /// allocate output vector per size of file
        bytevec_t result(fsize);
        
        /// start as you mean to go on
        ::lseek(descriptor, 0, SEEK_SET);
        
        /// unbuffered read directly from descriptor:
        if (::read(descriptor, &result[0], fsize) == -1) {
            imread_raise(CannotReadError,
                "fd_source_sink::full_data():",
                "read() returned -1", std::strerror(errno));
        }
        
        /// reset descriptor position before returning
        ::lseek(descriptor, orig, SEEK_SET);
        return result;
    }
    
    std::size_t fd_source_sink::size() const {
        detail::stat_t info = this->stat();
        return info.st_size * sizeof(byte);
    }
    
    void* fd_source_sink::readmap(std::size_t pageoffset) const {
        if (!mapped.get()) {
            detail::stat_t info = this->stat();
            std::size_t fsize = info.st_size * sizeof(byte);
            off_t offset = pageoffset ? pageoffset * ::getpagesize() : 0;
            /// NB. MAP_POPULATE doesn't work on OS X
            void* mapped_ptr = ::mmap(nullptr, fsize, PROT_READ,
                                                      MAP_PRIVATE,
                                                      descriptor,
                                                      offset);
            if (mapped_ptr == MAP_FAILED) {
                imread_raise(FileSystemError,
                    "error mapping file descriptor for reading:",
                    std::strerror(errno));
            }
            mapped = detail::mapped_t{ mapped_ptr, [fsize](void* mp) {
                ::munmap(mp, fsize);
            }};
        }
        return mapped.get();
    }
    
    std::string fd_source_sink::xattr(std::string const& name) const {
        filesystem::attribute::accessor_t accessor(descriptor, name);
        return accessor.get();
    }
    
    std::string fd_source_sink::xattr(std::string const& name, std::string const& value) const {
        filesystem::attribute::accessor_t accessor(descriptor, name);
        (value == filesystem::attribute::detail::nullstring) ? accessor.del() : accessor.set(value);
        return accessor.get();
    }
    
    int fd_source_sink::xattrcount() const {
        return filesystem::attribute::fdcount(descriptor);
    }
    
    filesystem::detail::stringvec_t fd_source_sink::xattrs() const {
        return filesystem::attribute::fdlist(descriptor);
    }
    
    int fd_source_sink::fd() const noexcept {
        return descriptor;
    }
    
    void fd_source_sink::fd(int fd) noexcept {
        if (fd_source_sink::check_descriptor(fd)) {
            descriptor = fd;
        }
    }
    
    filesystem::file fd_source_sink::fh() const noexcept {
        return filesystem::file(::fdopen(descriptor, "r+"));
    }
    
    void fd_source_sink::fh(FILE* fh) noexcept {
        int fd = ::fileno(fh);
        if (fd_source_sink::check_descriptor(fd)) {
            descriptor = fd;
        }
    }
    
    bool fd_source_sink::exists() const noexcept {
        try {
            this->stat();
        } catch (CannotReadError&) {
            return false;
        }
        return true;
    }
    
    bool fd_source_sink::is_open() const noexcept {
        return fd_source_sink::check_descriptor(descriptor);
    }
    
    int fd_source_sink::open(std::string const& spath,
                             filesystem::mode fmode) {
        if (fmode == filesystem::mode::WRITE) {
            descriptor = open_write(spath.c_str());
            if (!fd_source_sink::check_descriptor(descriptor)) {
                std::string open_error = std::strerror(errno);
                imread_raise(CannotWriteError, "descriptor open-to-write failure:",
                    FF("\t::open(\"%s\", O_WRONLY | O_FSYNC | O_CLOEXEC | O_CREAT | O_EXCL | O_TRUNC)", spath.c_str()),
                    FF("\treturned invalid descriptor: %i", descriptor),
                       "\tERROR MESSAGE IS:  ", std::strerror(errno),
                       "\tERROR MESSAGE WAS: ", open_error);
            }
        } else {
            descriptor = open_read(spath.c_str());
            if (!fd_source_sink::check_descriptor(descriptor)) {
                std::string open_error = std::strerror(errno);
                imread_raise(CannotReadError, "descriptor open-to-read failure:",
                    FF("\t::open(\"%s\", O_RDONLY | O_FSYNC | O_CLOEXEC)", spath.c_str()),
                    FF("\treturned invalid descriptor: %i", descriptor),
                       "\tERROR MESSAGE IS:  ", std::strerror(errno),
                       "\tERROR MESSAGE WAS: ", open_error);
            }
        }
        return descriptor;
    }
    
    int fd_source_sink::close() {
        using std::swap;
        int out = -1;
        mapped.reset(nullptr);
        if (fd_source_sink::check_descriptor(descriptor)) {
            if (::close(descriptor) != -1) {
                swap(out, descriptor);
            }
        }
        return out;
    }
    
    file_source_sink::file_source_sink(filesystem::mode fmode)
        :fd_source_sink(), md(fmode)
        {}
    
    file_source_sink::file_source_sink(char* cpath, filesystem::mode fmode)
        :fd_source_sink(), pth(cpath), md(fmode)
        {
            fd_source_sink::open(pth.str(), fmode);
        }
    
    file_source_sink::file_source_sink(char const* ccpath, filesystem::mode fmode)
        :fd_source_sink(), pth(ccpath), md(fmode)
        {
            fd_source_sink::open(pth.str(), fmode);
        }
    
    file_source_sink::file_source_sink(std::string& spath, filesystem::mode fmode)
        :file_source_sink(spath.c_str(), fmode)
        {}
    
    file_source_sink::file_source_sink(std::string const& cspath, filesystem::mode fmode)
        :file_source_sink(cspath.c_str(), fmode)
        {}
    
    file_source_sink::file_source_sink(filesystem::path const& ppath, filesystem::mode fmode)
        :fd_source_sink(), pth(ppath), md(fmode)
        {
            fd_source_sink::open(pth.str(), fmode);
        }
    
    filesystem::path const& file_source_sink::path() const {
        return pth;
    }
    
    bool file_source_sink::exists() const noexcept {
        return pth.exists();
    }
    
    filesystem::mode file_source_sink::mode(filesystem::mode m) {
        md = m;
        return md;
    }
    
    filesystem::mode file_source_sink::mode() const {
        return md;
    }
    
    fifo_source_sink::fifo_source_sink(filesystem::mode fmode)
        :fd_source_sink(), md(fmode)
        {}
    
    fifo_source_sink::fifo_source_sink(char* cpath, filesystem::mode fmode)
        :fd_source_sink(), pth(cpath), md(fmode)
        {
            if (!pth.exists()) { pth.makefifo(); }
            fifo_source_sink::open(pth.str(), fmode);
        }
    
    fifo_source_sink::fifo_source_sink(char const* ccpath, filesystem::mode fmode)
        :fd_source_sink(), pth(ccpath), md(fmode)
        {
            if (!pth.exists()) { pth.makefifo(); }
            fifo_source_sink::open(pth.str(), fmode);
        }
    
    fifo_source_sink::fifo_source_sink(std::string& spath, filesystem::mode fmode)
        :fifo_source_sink(spath.c_str(), fmode)
        {}
    
    fifo_source_sink::fifo_source_sink(std::string const& cspath, filesystem::mode fmode)
        :fifo_source_sink(cspath.c_str(), fmode)
        {}
    
    fifo_source_sink::fifo_source_sink(filesystem::path const& ppath, filesystem::mode fmode)
        :fd_source_sink(), pth(ppath), md(fmode)
        {
            if (!pth.exists()) { pth.makefifo(); }
            fifo_source_sink::open(pth.str(), fmode);
        }
    
    filesystem::path const& fifo_source_sink::path() const {
        return pth;
    }
    
    bool fifo_source_sink::exists() const noexcept {
        return pth.exists();
    }
    
    int fifo_source_sink::open(std::string const& spath,
                               filesystem::mode fmode) {
        if (fmode == filesystem::mode::WRITE) {
            descriptor = fifo_open_write(spath.c_str());
            if (descriptor < 0) {
                imread_raise(CannotWriteError, "FIFO descriptor open-to-write failure:",
                    FF("\t::open(\"%s\", O_WRONLY | O_CLOEXEC)", spath.c_str()),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        } else {
            descriptor = fifo_open_read(spath.c_str());
            if (descriptor < 0) {
                imread_raise(CannotReadError, "FIFO descriptor open-to-read failure:",
                    FF("\t::open(\"%s\", O_RDONLY | O_CLOEXEC)", spath.c_str()),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        }
        return descriptor;
    }
    
    filesystem::mode fifo_source_sink::mode(filesystem::mode m) {
        md = m;
        return md;
    }
    
    filesystem::mode fifo_source_sink::mode() const {
        return md;
    }
    
    FileSource::FileSource()
        :file_source_sink()
        {}
    
    FileSource::FileSource(char* cpath)
        :file_source_sink(cpath)
        {}
    
    FileSource::FileSource(char const* ccpath)
        :file_source_sink(ccpath)
        {}
    
    FileSource::FileSource(std::string& spath)
        :file_source_sink(spath)
        {}
    
    FileSource::FileSource(std::string const& cspath)
        :file_source_sink(cspath)
        {}
    
    FileSource::FileSource(filesystem::path const& ppath)
        :file_source_sink(ppath)
        {}
    
    FileSink::FileSink()
        :file_source_sink(filesystem::mode::WRITE)
        {}
    
    FileSink::FileSink(char* cpath)
        :file_source_sink(cpath, filesystem::mode::WRITE)
        {}
    
    FileSink::FileSink(char const* ccpath)
        :file_source_sink(ccpath, filesystem::mode::WRITE)
        {}
    
    FileSink::FileSink(std::string& spath)
        :file_source_sink(spath, filesystem::mode::WRITE)
        {}
    
    FileSink::FileSink(std::string const& cspath)
        :file_source_sink(cspath, filesystem::mode::WRITE)
        {}
    
    FileSink::FileSink(filesystem::path const& ppath)
        :file_source_sink(ppath, filesystem::mode::WRITE)
        {}
    
    FIFOSource::FIFOSource()
        :fifo_source_sink()
        {}
    
    FIFOSource::FIFOSource(char* cpath)
        :fifo_source_sink(cpath)
        {}
    
    FIFOSource::FIFOSource(char const* ccpath)
        :fifo_source_sink(ccpath)
        {}
    
    FIFOSource::FIFOSource(std::string& spath)
        :fifo_source_sink(spath)
        {}
    
    FIFOSource::FIFOSource(std::string const& cspath)
        :fifo_source_sink(cspath)
        {}
    
    FIFOSource::FIFOSource(filesystem::path const& ppath)
        :fifo_source_sink(ppath)
        {}
    
    FIFOSink::FIFOSink()
        :fifo_source_sink(filesystem::mode::WRITE)
        {}
    
    FIFOSink::FIFOSink(char* cpath)
        :fifo_source_sink(cpath, filesystem::mode::WRITE)
        {}
    
    FIFOSink::FIFOSink(char const* ccpath)
        :fifo_source_sink(ccpath, filesystem::mode::WRITE)
        {}
    
    FIFOSink::FIFOSink(std::string& spath)
        :fifo_source_sink(spath, filesystem::mode::WRITE)
        {}
    
    FIFOSink::FIFOSink(std::string const& cspath)
        :fifo_source_sink(cspath, filesystem::mode::WRITE)
        {}
    
    FIFOSink::FIFOSink(filesystem::path const& ppath)
        :fifo_source_sink(ppath, filesystem::mode::WRITE)
        {}
    
}
