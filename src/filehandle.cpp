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
#include <libimread/filehandle.hh>

namespace im {
    
    std::size_t handle_source_sink::seek_absolute(std::size_t pos) { std::fseek(handle, pos, SEEK_SET); return std::ftell(handle); }
    std::size_t handle_source_sink::seek_relative(int delta) { std::fseek(handle, delta, SEEK_CUR); return std::ftell(handle); }
    std::size_t handle_source_sink::seek_end(int delta) { std::fseek(handle, delta, SEEK_END); return std::ftell(handle); }
    
    std::size_t handle_source_sink::read(byte* buffer, std::size_t n) {
        std::size_t out = std::fread(buffer, sizeof(byte), n, handle);
        if (out == -1) {
            imread_raise(CannotReadError,
                "std::fread() returned -1",
                std::strerror(errno));
        }
        return out;
    }
    
    std::size_t handle_source_sink::write(const void* buffer, std::size_t n) {
        std::size_t out = std::fwrite(buffer, sizeof(byte), n, handle);
        if (out == -1) {
            imread_raise(CannotWriteError,
                "std::fwrite() returned -1",
                std::strerror(errno));
        }
        return out;
    }
    
    std::size_t handle_source_sink::write(std::vector<byte> const& bv) {
        return this->write(
            static_cast<const void*>(&bv[0]),
            bv.size());
    }
    
    detail::stat_t handle_source_sink::stat() const {
        detail::stat_t info;
        if (::fstat(::fileno(handle), &info) == -1) {
            imread_raise(CannotReadError,
                "::fstat(::fileno()) returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    void handle_source_sink::flush() { std::fflush(handle); }
    
    std::vector<byte> handle_source_sink::full_data() {
        /// grab stat struct and store initial seek position
        detail::stat_t info = this->stat();
        std::size_t fsize = info.st_size * sizeof(byte);
        std::size_t orig = std::ftell(handle);
        
        /// allocate output vector per size of file
        std::vector<byte> result(fsize);
        
        /// start as you mean to go on
        std::rewind(handle);
        
        /// read directly from filehandle:
        if (std::fread(&result[0], sizeof(byte), fsize, handle) == -1) {
            imread_raise(CannotReadError,
                "handle_source_sink::full_data():",
                "std::fread() returned -1", std::strerror(errno));
        }
        
        /// reseek to the streams' original position
        std::fseek(handle, orig, SEEK_SET);
        return result;
    }
    
    std::size_t handle_source_sink::size() {
        detail::stat_t info = this->stat();
        return info.st_size * sizeof(byte);
    }
    
    int handle_source_sink::fd() const noexcept {
        return ::fileno(handle);
    }
    
    void handle_source_sink::fd(int descriptor) noexcept {
        FILE* fh = ::fdopen(descriptor, "r+");
        if (fh != nullptr) { handle = fh; }
    }
    
    FILE* handle_source_sink::fh() const noexcept {
        return handle;
    }
    
    void handle_source_sink::fh(FILE* fh) noexcept {
        if (fh != nullptr) {
            handle = fh;
            external = true;
        }
    }
    
    bool handle_source_sink::exists() const noexcept {
        try {
            this->stat();
        } catch (CannotReadError const& e) {
            return false;
        }
        return true;
    }
    
    FILE* handle_source_sink::open(char* cpath, filesystem::mode fmode) {
        handle = std::fopen(cpath, fmode == filesystem::mode::READ ? "r+" : "w");
        if (!handle) {
            imread_raise(CannotReadError, "filehandle open failure:",
                FF("\tstd::fopen(\"%s\", \"%s\")", cpath,
                        fmode == filesystem::mode::READ ? "r+" : "w"),
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
                    "error closing filehandle:",
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
