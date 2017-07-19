/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_GZIO_HH_
#define LIBIMREAD_GZIO_HH_

#include <fcntl.h>
#include <zlib.h>

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
#include <libimread/store.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using gzhandle_t = gzFile;
        /// -- a camel-cased typename, AND a typedef that obfuscates a pointer ...
        /// HOW COULD IT POSSIBLY HAVE LESS AESTHETICALLY PLEASING QUALITIES, I ask you
    }
    
    class gzio_source_sink : public byte_source, public byte_sink, public store::xattrmap {
        
        protected:
            static constexpr int kReadFlags             = O_RDONLY | O_NONBLOCK;
            static constexpr int kWriteFlags            = O_WRONLY | O_NONBLOCK | O_CREAT | O_EXCL | O_TRUNC;
            static constexpr int kWriteCreateMask       = 0644;
            
            static constexpr char kOriginalSize[]       = "im:original_size";
            static constexpr char kUncompressedSize[]   = "im:uncompressed_size";
            
            static int open_read(char const* p);
            static int open_write(char const* p, int mask = kWriteCreateMask);
        
        public:
            static std::size_t max_descriptor_count();
            static std::size_t max_descriptor_count(std::size_t);
        
        public:
            gzio_source_sink();
            gzio_source_sink(detail::gzhandle_t gz);
            gzio_source_sink(int fd);
            
            virtual ~gzio_source_sink();
            
            virtual bool can_seek() const noexcept override;
            virtual bool can_store() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t pos) override;
            virtual std::size_t seek_relative(int delta) override;
            virtual std::size_t seek_end(int delta) override;
            
            virtual std::size_t read(byte* buffer, std::size_t n) override;
            virtual std::size_t size() const override;
            virtual std::size_t write(const void* buffer, std::size_t n) override;
            virtual std::size_t write(bytevec_t const&) override;
            virtual detail::stat_t stat() const;
            virtual void flush() override;
            
            virtual void* readmap(std::size_t pageoffset = 0) const override;
            
            /// Filesystem extended attribute (“xattr”) access
            virtual std::string xattr(std::string const&) const override;
            virtual std::string xattr(std::string const&, std::string const&) const override;
            virtual int xattrcount() const override;
            virtual filesystem::detail::stringvec_t xattrs() const override;
            
            /// implementation-specific xattr access
            std::size_t original_byte_size() const;
            std::size_t original_byte_size(std::size_t) const;
            std::size_t uncompressed_byte_size() const;
            std::size_t uncompressed_byte_size(std::size_t) const;
            float       compression_ratio() const;
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            
            virtual bool exists() const noexcept;
            virtual int open(std::string const& spath, filesystem::mode fmode = filesystem::mode::READ);
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
                source(char* cpath);
                source(char const* ccpath);
                source(std::string& spath);
                source(std::string const& cspath);
                source(filesystem::path const& ppath);
                
            public:
                /// for im::gzio::source::load(…)
                DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(gzio::source);
        };
        
        class sink : public gzfile_source_sink {
            
            public:
                sink();
                sink(char* cpath);
                sink(char const* ccpath);
                sink(std::string& spath);
                sink(std::string const& cspath);
                sink(filesystem::path const& ppath);
                
            public:
                /// for im::gzio::sink::load(…)
                DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(gzio::sink);
        };
        
    } /* namespace gzio */
    
} /* namespace im */

#endif /// LIBIMREAD_GZIO_HH_
