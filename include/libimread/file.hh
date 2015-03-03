// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#ifndef LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#if defined(_MSC_VER)
    #include <io.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
#endif

#include <errno.h>
#ifndef O_BINARY
const int O_BINARY = 0;
#endif

#include <cstring>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>

namespace im {

    class fd_source_sink : public byte_source, public byte_sink {
        
        public:
            fd_source_sink() {}
            fd_source_sink(int fd)
                :fd_(fd)
                { }
            
            ~fd_source_sink() {
                ::close(fd_);
            }
            
            virtual size_t read(byte* buffer, size_t n) {
                return ::read(fd_, buffer, n);
            }
            
            virtual bool can_seek() const { return true; }
            virtual size_t seek_absolute(size_t pos) { return ::lseek(fd_, pos, SEEK_SET); }
            virtual size_t seek_relative(int delta) { return ::lseek(fd_, delta, SEEK_CUR); }
            virtual size_t seek_end(int delta) { return ::lseek(fd_, delta, SEEK_END); }
            
            virtual size_t write(const byte* buffer, size_t n) {
                return ::write(fd_, buffer, n);
            }
            
            virtual void set_fd(int fd) { fd_ = fd; }
            
        private:
            int fd_;
    };
    
    class file_source_sink : public fd_source_sink {
        private:
            std::unique_ptr<char[]> pth;
        
        public:
            file_source_sink(char *cpath)
                :fd_source_sink()
                {
                    int fd = ::open(cpath, O_RDONLY | O_BINARY);
                    if (fd < 0) {
                        throw CannotReadError("Failed to read file");
                    }
                    this->set_fd(fd);
                    pth = std::make_unique<char[]>(std::strlen(cpath)+1);
                    std::strcpy(pth.get(), cpath);
                }
            
            file_source_sink(const char *ccpath)
                :file_source_sink(const_cast<char *>(ccpath))
                {}
            
            file_source_sink(std::string &spath)
                :file_source_sink(spath.c_str())
                {}
            
            file_source_sink(const std::string &cspath)
                :file_source_sink(cspath.c_str())
                {}
            
            char *path() const { return pth.get(); }
    };
    

}

#endif // LPC_FILE_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
