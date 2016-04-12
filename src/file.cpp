/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/file.hh>

namespace im {
    
    constexpr int fd_source_sink::READ_FLAGS;
    constexpr int fd_source_sink::WRITE_FLAGS;
    constexpr int fd_source_sink::WRITE_CREATE_MASK;
    
    int fd_source_sink::open_read(char* p) const {
        return ::open(p, READ_FLAGS);
    }
    
    int fd_source_sink::open_write(char* p, int mask) const {
        return ::open(p, WRITE_FLAGS, mask);
    }
    
    std::size_t fd_source_sink::seek_absolute(std::size_t pos) { return ::lseek(descriptor, pos, SEEK_SET); }
    std::size_t fd_source_sink::seek_relative(int delta) { return ::lseek(descriptor, delta, SEEK_CUR); }
    std::size_t fd_source_sink::seek_end(int delta) { return ::lseek(descriptor, delta, SEEK_END); }
    
    std::size_t fd_source_sink::read(byte* buffer, std::size_t n) {
        int out = ::read(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotReadError,
                "read() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t fd_source_sink::write(const void* buffer, std::size_t n) {
        int out = ::write(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "write() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    std::size_t fd_source_sink::write(std::vector<byte> const& bv) {
        return this->write(static_cast<const void*>(&bv[0]),
                           bv.size());
    }
    
    detail::stat_t fd_source_sink::stat() const {
        detail::stat_t info;
        if (::fstat(descriptor, &info) == -1) {
            imread_raise(CannotReadError,
                "fstat() returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    std::vector<byte> fd_source_sink::full_data() {
        /// grab stat struct and store initial seek position
        detail::stat_t info = this->stat();
        std::size_t orig = ::lseek(descriptor, 0, SEEK_CUR);
        
        /// allocate output vector per size of file
        std::vector<byte> res(info.st_size * sizeof(byte));
        
        /// start as you mean to go on
        ::lseek(descriptor, 0, SEEK_SET);
        
        /// unbuffered read directly from descriptor:
        if (::read(descriptor, &res[0], res.size()) == -1) {
            imread_raise(CannotReadError,
                "error in full_data(): read() returned -1",
                std::strerror(errno));
        }
        
        /// reset descriptor position before returning
        ::lseek(descriptor, orig, SEEK_SET);
        return res;
    }
    
    int fd_source_sink::fd() const noexcept {
        return descriptor;
    }
    
    void fd_source_sink::fd(int fd) noexcept {
        descriptor = fd;
    }
    
    bool fd_source_sink::exists() const noexcept {
        try {
            this->stat();
        } catch (const CannotReadError& e) {
            return false;
        }
        return true;
    }
    
    int fd_source_sink::open(char* cpath,
                             filesystem::mode fmode) {
        if (fmode == filesystem::mode::WRITE) {
            descriptor = open_write(cpath);
            if (descriptor < 0) {
                imread_raise(CannotWriteError, "file open-to-write failure:",
                    FF("\t::open(\"%s\", O_WRONLY | O_FSYNC | O_CREAT | O_EXCL | O_TRUNC)", cpath),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        } else {
            descriptor = open_read(cpath);
            if (descriptor < 0) {
                imread_raise(CannotReadError, "file open-to-read failure:",
                    FF("\t::open(\"%s\", O_RDONLY | O_FSYNC)", cpath),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        }
        return descriptor;
    }
    
    int fd_source_sink::close() {
        using std::swap;
        int out = -1;
        if (descriptor > 0) {
            if (::close(descriptor) == -1) {
                imread_raise(FileSystemError,
                    "error while closing file descriptor:",
                    std::strerror(errno));
            }
            swap(out, descriptor);
        }
        return out;
    }
    
    bool file_source_sink::exists() const noexcept { return pth.exists(); }
    
}
