/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FILE_HH_
#define LIBIMREAD_FILE_HH_

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cerrno>
#include <sstream>
#include <cstring>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>

namespace im {
    
    class fd_source_sink : public byte_source, public byte_sink {
        
        public:
            fd_source_sink() {}
            fd_source_sink(int fd)
                :descriptor(fd)
                { }
            
            virtual ~fd_source_sink() {
                ::close(descriptor);
            }
            
            virtual std::size_t read(byte *buffer, std::size_t n) {
                return ::read(descriptor, buffer, n);
            }
            
            virtual bool can_seek() const { return true; }
            virtual std::size_t seek_absolute(std::size_t pos) { return ::lseek(descriptor, pos, SEEK_SET); }
            virtual std::size_t seek_relative(int delta) { return ::lseek(descriptor, delta, SEEK_CUR); }
            virtual std::size_t seek_end(int delta) { return ::lseek(descriptor, delta, SEEK_END); }
            
            virtual std::size_t write(const void *buffer, std::size_t n) {
                return ::write(descriptor, buffer, n);
            }
            
            virtual std::vector<byte> full_data() {
                std::size_t orig = this->seek_relative(0);
                
                struct stat info;
                int result = ::fstat(descriptor, &info);
                if (result == -1) {
                    imread_raise(CannotReadError,
                        "fstat() returned -1",
                        std::strerror(errno));
                }
                
                std::vector<byte> res(info.st_size * sizeof(byte));
                this->seek_absolute(0);
                this->read(&res[0], res.size());
                this->seek_absolute(orig);
                return res;
            }
            
            virtual int fd() const { return descriptor; }
            virtual void fd(int fd) { descriptor = fd; }
            
        private:
            int descriptor;
    };
    
    class file_source_sink : public fd_source_sink {
        private:
            filesystem::path pth;
            filesystem::mode md;
            
            static constexpr int READ_FLAGS = O_RDONLY | O_NONBLOCK;
            static constexpr int WRITE_FLAGS = O_CREAT | O_WRONLY | O_TRUNC | O_EXLOCK | O_SYMLINK;
            int open_read(char *p) const { return ::open(p, READ_FLAGS); }
            int open_write(char *p, int m=0644) const { return ::open(p, WRITE_FLAGS, m); }
        
        public:
            file_source_sink(filesystem::mode fmode = filesystem::mode::READ)
                :fd_source_sink(), md(fmode)
                {}
            
            file_source_sink(char *cpath, filesystem::mode fmode = filesystem::mode::READ)
                :fd_source_sink(), pth(cpath), md(fmode)
                {
                    int descriptor = -1;
                    if (md == filesystem::mode::READ) {
                        descriptor = open_read(cpath);
                    } else if (md == filesystem::mode::WRITE) {
                        descriptor = open_write(cpath);
                    }
                    if (descriptor < 0) {
                        imread_raise(CannotReadError, "file read failure:",
                            FF("\t::open(\"%s\", %s)", cpath, ((md == filesystem::mode::READ)
                                        ? "O_RDONLY | O_NONBLOCK"
                                        : "O_CREAT | O_WRONLY | O_TRUNC | O_EXLOCK | O_SYMLINK")),
                            FF("\treturned negative value: %i", descriptor),
                               "\tERROR MESSAGE IS: ", std::strerror(errno));
                    }
                    this->fd(descriptor);
                }
            
            file_source_sink(const char *ccpath, filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(const_cast<char *>(ccpath), fmode)
                {}
            
            file_source_sink(std::string &spath, filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(spath.c_str(), fmode)
                {}
            
            file_source_sink(const std::string &cspath, filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(cspath.c_str(), fmode)
                {}
            
            file_source_sink(const filesystem::path &ppath, filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(ppath.c_str(), fmode)
                {}
            
            filesystem::path &path() { return pth; }
            bool exists() const;
            
            filesystem::mode mode() { return md; }
            void mode(filesystem::mode m) { md = m; }
    };
    
    class FileSource : public file_source_sink {
        public:
            FileSource()
                :file_source_sink()
                {}
            FileSource(char *cpath)
                :file_source_sink(cpath)
                {}
            FileSource(const char *ccpath)
                :file_source_sink(ccpath)
                {}
            FileSource(std::string &spath)
                :file_source_sink(spath)
                {}
            FileSource(const std::string &cspath)
                :file_source_sink(cspath)
                {}
            FileSource(const filesystem::path &ppath)
                :file_source_sink(ppath)
                {}
    };
    
    class FileSink : public file_source_sink {
        public:
            FileSink()
                :file_source_sink(filesystem::mode::WRITE)
                {}
            FileSink(char *cpath)
                :file_source_sink(cpath, filesystem::mode::WRITE)
                {}
            FileSink(const char *ccpath)
                :file_source_sink(ccpath, filesystem::mode::WRITE)
                {}
            FileSink(std::string &spath)
                :file_source_sink(spath, filesystem::mode::WRITE)
                {}
            FileSink(const std::string &cspath)
                :file_source_sink(cspath, filesystem::mode::WRITE)
                {}
            FileSink(const filesystem::path &ppath)
                :file_source_sink(ppath, filesystem::mode::WRITE)
                {}
    };
    

}

#endif /// LIBIMREAD_FILE_HH_
