/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_GZIO_HH_
#define LIBIMREAD_GZIO_HH_

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
        using gzhandle_t = gzFile;
        /// -- a camel-cased typename and a typedef that obfuscates a pointer ...
        /// HOW COULD IT POSSIBLY HAVE LESS AESTHETICALLY PLEASING QUALITIES, I ask you
    }
    
    class gzio_source_sink : public byte_source, public byte_sink {
        
        protected:
            static constexpr int READ_FLAGS = O_RDONLY | O_NONBLOCK;
            static constexpr int WRITE_FLAGS = O_WRONLY | O_NONBLOCK | O_CREAT | O_EXCL | O_TRUNC;
            static constexpr int WRITE_CREATE_MASK = 0644;
            
            int open_read(char* p) const;
            int open_write(char* p, int mask = WRITE_CREATE_MASK) const;
        
        public:
            gzio_source_sink();
            gzio_source_sink(detail::gzhandle_t gz);
            gzio_source_sink(int fd);
            
            virtual ~gzio_source_sink();
            
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t pos);
            virtual std::size_t seek_relative(int delta);
            virtual std::size_t seek_end(int delta);
            
            virtual std::size_t read(byte* buffer, std::size_t n);
            // virtual std::vector<byte> full_data();
            virtual std::size_t size() const;
            virtual std::size_t write(const void* buffer, std::size_t n);
            virtual std::size_t write(std::vector<byte> const& bv);
            virtual detail::stat_t stat() const;
            virtual void flush();
            
            virtual void* readmap(std::size_t pageoffset = 0) const;
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            // virtual filesystem::file fh() const noexcept;
            // virtual void fh(FILE* fh) noexcept;
            
            virtual bool exists() const noexcept;
            virtual int open(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            virtual int close();
            
        private:
            int descriptor = -1;
            mutable detail::gzhandle_t gzhandle;
            bool external = false;
    };
    
    class gzfile_source_sink : public gzio_source_sink {
        private:
            filesystem::path pth;
            filesystem::mode md;
        
        public:
            gzfile_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            gzfile_source_sink(char* cpath,
                               filesystem::mode fmode = filesystem::mode::READ);
            gzfile_source_sink(char const* ccpath,
                               filesystem::mode fmode = filesystem::mode::READ);
            gzfile_source_sink(std::string& spath,
                               filesystem::mode fmode = filesystem::mode::READ);
            gzfile_source_sink(std::string const& cspath,
                               filesystem::mode fmode = filesystem::mode::READ);
            gzfile_source_sink(filesystem::path const& ppath,
                               filesystem::mode fmode = filesystem::mode::READ);
            
            virtual ~gzfile_source_sink();
            
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            filesystem::mode mode(filesystem::mode m);
            filesystem::mode mode() const;
    };
    
    namespace gzio {
        
        /// NB. Acronym-pronouncers can phonetically call this “Gizzy-O”
        
        class source : public gzfile_source_sink {
            public:
                source();
                // source(FILE* fh);
                source(char* cpath);
                source(char const* ccpath);
                source(std::string& spath);
                source(std::string const& cspath);
                source(filesystem::path const& ppath);
        };
        
        class sink : public gzfile_source_sink {
            public:
                sink();
                // sink(FILE* fh);
                sink(char* cpath);
                sink(char const* ccpath);
                sink(std::string& spath);
                sink(std::string const& cspath);
                sink(filesystem::path const& ppath);
        };
        
    } /* namespace gzio */
    
} /* namespace im */

#endif /// LIBIMREAD_GZIO_HH_
