/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/file.hh>

namespace im {
    
    constexpr int fd_source_sink::READ_FLAGS;
    constexpr int fd_source_sink::WRITE_FLAGS;
    constexpr int fd_source_sink::WRITE_CREATE_MASK;
    
    std::size_t fd_source_sink::read(byte *buffer, std::size_t n) {
        int out = ::read(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotReadError,
                "read() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t fd_source_sink::write(const void *buffer, std::size_t n) {
        int out = ::write(descriptor, buffer, n);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "write() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    std::size_t fd_source_sink::write(const std::vector<byte> &bv) {
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
        std::size_t orig = ::lseek(descriptor, 0, SEEK_CUR);
        detail::stat_t info = this->stat();
        
        std::vector<byte> res(info.st_size * sizeof(byte));
        ::lseek(descriptor, 0, SEEK_SET);
        if (::read(descriptor, &res[0], res.size()) == -1) {
            imread_raise(CannotReadError,
                "error in full_data(): read() returned -1",
                std::strerror(errno));
        }
        ::lseek(descriptor, orig, SEEK_SET);
        return res;
    }
    
    int fd_source_sink::fd() const {
        return descriptor;
    }
    
    void fd_source_sink::fd(int fd) {
        descriptor = fd;
    }
    
    bool fd_source_sink::exists() const noexcept {
        try {
            this->stat();
        } catch (const CannotReadError &e) {
            return false;
        }
        return true;
    }
    
    int fd_source_sink::open(char *cpath,
                             filesystem::mode fmode) {
        if (fmode == filesystem::mode::WRITE) {
            descriptor = open_write(cpath);
        } else {
            descriptor = open_read(cpath);
        }
        if (descriptor < 0) {
            imread_raise(CannotReadError, "file read failure:",
                FF("\t::open(\"%s\", %s)", cpath, ((fmode == filesystem::mode::READ)
                            ? "O_RDONLY | O_NONBLOCK"
                            : "O_CREAT | O_WRONLY | O_TRUNC | O_EXLOCK | O_SYMLINK")),
                FF("\treturned negative value: %i", descriptor),
                   "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        return descriptor;
    }
    
    int fd_source_sink::close() {
        int out = -1;
        if (descriptor > 0) {
            if (::close(descriptor) == -1) {
                imread_raise(FileSystemError,
                    "error while closing file descriptor:",
                    std::strerror(errno));
            }
            std::swap(out, descriptor);
        }
        return out;
    }
    
    bool file_source_sink::exists() const noexcept { return pth.exists(); }
    
}
