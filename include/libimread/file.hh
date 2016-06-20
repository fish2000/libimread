/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FILE_HH_
#define LIBIMREAD_FILE_HH_

#include <fcntl.h>
#include <cstdio>
#include <vector>
#include <memory>
#include <functional>
#include <utility>
#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/seekable.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class fd_source_sink : public byte_source, public byte_sink {
        
        protected:
            static constexpr int READ_FLAGS = O_RDONLY | O_NONBLOCK;
            static constexpr int WRITE_FLAGS = O_WRONLY | O_NONBLOCK | O_CREAT | O_EXCL | O_TRUNC;
            static constexpr int WRITE_CREATE_MASK = 0644;
            
            int open_read(char* p) const;
            int open_write(char* p, int mask = WRITE_CREATE_MASK) const;
        
        public:
            fd_source_sink();
            fd_source_sink(int fd);
            
            virtual ~fd_source_sink();
            
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t pos);
            virtual std::size_t seek_relative(int delta);
            virtual std::size_t seek_end(int delta);
            
            virtual std::size_t read(byte* buffer, std::size_t n);
            virtual std::vector<byte> full_data();
            virtual std::size_t size() const;
            virtual std::size_t write(const void* buffer, std::size_t n);
            virtual std::size_t write(std::vector<byte> const& bv);
            virtual detail::stat_t stat() const;
            virtual void flush();
            
            virtual void* readmap(std::size_t pageoffset = 0) const;
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            virtual filesystem::file fh() const noexcept;
            virtual void fh(FILE* fh) noexcept;
            
            virtual bool exists() const noexcept;
            virtual int open(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            virtual int close();
            
        private:
            int descriptor = -1;
            mutable detail::mapped_t mapped;
    };
    
    class file_source_sink : public fd_source_sink {
        private:
            filesystem::path pth;
            filesystem::mode md;
        
        public:
            file_source_sink(filesystem::mode fmode = filesystem::mode::READ)
                :fd_source_sink(), md(fmode)
                {}
            
            file_source_sink(char* cpath,
                             filesystem::mode fmode = filesystem::mode::READ)
                :fd_source_sink(), pth(cpath), md(fmode)
                {
                    fd_source_sink::open(cpath, fmode);
                }
            
            file_source_sink(char const* ccpath,
                             filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(const_cast<char*>(ccpath), fmode)
                {}
            
            file_source_sink(std::string& spath,
                             filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(spath.c_str(), fmode)
                {}
            
            file_source_sink(std::string const& cspath,
                             filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(cspath.c_str(), fmode)
                {}
            
            file_source_sink(filesystem::path const& ppath,
                             filesystem::mode fmode = filesystem::mode::READ)
                :file_source_sink(ppath.c_str(), fmode)
                {}
            
            virtual ~file_source_sink() { fd_source_sink::close(); }
            
            filesystem::path const& path() const { return pth; }
            virtual bool exists() const noexcept override;
            
            void mode(filesystem::mode m) { md = m; }
            filesystem::mode mode() const { return md; }
    };
    
    class FileSource : public file_source_sink {
        public:
            FileSource()
                :file_source_sink()
                {}
            FileSource(char* cpath)
                :file_source_sink(cpath)
                {}
            FileSource(char const* ccpath)
                :file_source_sink(ccpath)
                {}
            FileSource(std::string& spath)
                :file_source_sink(spath)
                {}
            FileSource(std::string const& cspath)
                :file_source_sink(cspath)
                {}
            FileSource(filesystem::path const& ppath)
                :file_source_sink(ppath)
                {}
    };
    
    class FileSink : public file_source_sink {
        public:
            FileSink()
                :file_source_sink(filesystem::mode::WRITE)
                {}
            FileSink(char* cpath)
                :file_source_sink(cpath, filesystem::mode::WRITE)
                {}
            FileSink(char const* ccpath)
                :file_source_sink(ccpath, filesystem::mode::WRITE)
                {}
            FileSink(std::string& spath)
                :file_source_sink(spath, filesystem::mode::WRITE)
                {}
            FileSink(std::string const& cspath)
                :file_source_sink(cspath, filesystem::mode::WRITE)
                {}
            FileSink(filesystem::path const& ppath)
                :file_source_sink(ppath, filesystem::mode::WRITE)
                {}
    };
    

}

#endif /// LIBIMREAD_FILE_HH_
