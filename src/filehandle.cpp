/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/filehandle.hh>

namespace im {
    
    std::size_t handle_source_sink::seek_absolute(std::size_t pos) { return std::fseek(handle, pos, SEEK_SET); }
    std::size_t handle_source_sink::seek_relative(int delta) { return std::fseek(handle, delta, SEEK_CUR); }
    std::size_t handle_source_sink::seek_end(int delta) { return std::fseek(handle, delta, SEEK_END); }
    
    std::size_t handle_source_sink::read(byte* buffer, std::size_t n) {
        std::size_t out = std::fread(buffer, 1, n, handle);
        if (out == -1) {
            imread_raise(CannotReadError,
                "read() returned -1",
                std::strerror(errno));
        }
        return out;
    }
    
    std::size_t handle_source_sink::write(const void* buffer, std::size_t n) {
        std::size_t out = std::fwrite(buffer, 1, n, handle);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "write() returned -1",
                std::strerror(errno));
        }
        return out;
    }
    std::size_t handle_source_sink::write(std::vector<byte> const& bv) {
        return this->write(static_cast<const void*>(&bv[0]),
                           bv.size());
    }
    
    detail::stat_t handle_source_sink::stat() const {
        detail::stat_t info;
        if (::fstat(::fileno(handle), &info) == -1) {
            imread_raise(CannotReadError,
                "fstat() returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    std::vector<byte> handle_source_sink::full_data() {
        /// grab stat struct and store initial seek position
        detail::stat_t info = this->stat();
        std::size_t orig = std::fseek(handle, 0, SEEK_CUR);
        
        /// allocate output vector per size of file
        std::vector<byte> res(info.st_size * sizeof(byte));
        
        /// start as you mean to go on
        std::fseek(handle, 0, SEEK_SET);
        
        /// unbuffered read directly from file handle:
        if (std::fread(&res[0], 1, res.size(), handle) == -1) {
            imread_raise(CannotReadError,
                "error in full_data(): read() returned -1",
                std::strerror(errno));
        }
        
        /// reseek to the streams' original position
        std::fseek(handle, orig, SEEK_SET);
        return res;
    }
    
    int handle_source_sink::fd() const noexcept {
        return ::fileno(handle);
    }
    
    void handle_source_sink::fd(int fd) noexcept {
        handle = ::fdopen(fd, "r+");
    }
    
    FILE* handle_source_sink::fh() const noexcept {
        return handle;
    }
    
    void handle_source_sink::fh(FILE* fh) noexcept {
        handle = fh;
        external = true;
    }
    
    bool handle_source_sink::exists() const noexcept {
        try {
            this->stat();
        } catch (const CannotReadError& e) {
            return false;
        }
        return true;
    }
    
    FILE* handle_source_sink::open(char* cpath,
                                 filesystem::mode fmode) {
        // if (fmode == filesystem::mode::WRITE) {
        //     handle = std::fopen(cpath, "r+");
        //     if (!handle) {
        //         imread_raise(CannotWriteError, "filehandle open-to-write failure:",
        //             FF("\tstd::fopen(\"%s\", \"r+\")", cpath),
        //                "\treturned nullptr value",
        //                "\tERROR MESSAGE IS: ", std::strerror(errno));
        //     }
        // } else {
        //     handle = std::fopen(cpath, "r+");
        //     if (!handle) {
        //         imread_raise(CannotReadError, "filehandle open-to-read failure:",
        //             FF("\tstd::fopen(\"%s\", \"r+\")", cpath),
        //                "\treturned nullptr value",
        //                "\tERROR MESSAGE IS: ", std::strerror(errno));
        //     }
        // }
        handle = std::fopen(cpath, "r+");
        if (!handle) {
            imread_raise(CannotReadError, "filehandle open failure:",
                FF("\tstd::fopen(\"%s\", \"r+\")", cpath),
                    "\treturned nullptr value",
                    "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        return handle;
    }
    
    FILE* handle_source_sink::close() {
        using std::swap;
        FILE* out = nullptr;
        if (handle) {
            if (std::fclose(handle) == -1) {
                imread_raise(FileSystemError,
                    "error while closing file hanlde:",
                    std::strerror(errno));
            }
            swap(out, handle);
        }
        return out;
    }
    
    filehandle_source_sink::filehandle_source_sink(FILE* fh, filesystem::mode fmode)
        :handle_source_sink(fh), pth(::fileno(fh)), md(fmode)
        {}
    
    filehandle_source_sink::filehandle_source_sink(char* cpath, filesystem::mode fmode)
        :handle_source_sink(), pth(cpath), md(fmode)
        {
            handle_source_sink::open(cpath, fmode);
        }
    
    bool filehandle_source_sink::exists() const noexcept { return pth.exists(); }
    
}
