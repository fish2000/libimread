/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FILEHANDLE_HH_
#define LIBIMREAD_FILEHANDLE_HH_

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

namespace capnp {
    class MessageBuilder;
}

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class handle_source_sink : public byte_source, public byte_sink, public store::xattrmap {
        
        public:
            handle_source_sink();
            handle_source_sink(FILE* fh);
            
            virtual ~handle_source_sink();
            
            /// im::seekable methods
            virtual bool can_seek() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t pos) override;
            virtual std::size_t seek_relative(int delta) override;
            virtual std::size_t seek_end(int delta) override;
            
            /// im::byte_source and im::byte_sink methods
            virtual std::size_t read(byte* buffer, std::size_t n) override;
            virtual std::vector<byte> full_data() override;
            virtual std::size_t size() const override;
            virtual std::size_t write(const void* buffer, std::size_t n) override;
            virtual std::size_t write(std::vector<byte> const&) override;
            virtual std::size_t write(capnp::MessageBuilder& builder);
            virtual detail::stat_t stat() const;
            virtual void flush() override;
            
            virtual void* readmap(std::size_t pageoffset = 0) const override;
            
            /// Filesystem extended attribute (“xattr”) access
            virtual std::string xattr(std::string const&) const override;
            virtual std::string xattr(std::string const&, std::string const&) const override;
            virtual int xattrcount() const override;
            virtual filesystem::detail::stringvec_t xattrs() const override;
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            virtual FILE* fh() const noexcept;
            virtual void fh(FILE* fh) noexcept;
            
            virtual bool exists() const noexcept;
            virtual FILE* open(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            virtual FILE* close();
            
        private:
            mutable FILE* handle = nullptr;
            mutable detail::mapped_t mapped;
            bool external = false;
    };
    
    class filehandle_source_sink : public handle_source_sink {
        private:
            filesystem::path pth;
            filesystem::mode md;
        
        public:
            filehandle_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(FILE* fh, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(char const* ccpath, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(std::string& spath, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(std::string const& cspath, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(filesystem::path const& ppath, filesystem::mode fmode = filesystem::mode::READ);
            
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            filesystem::mode mode(filesystem::mode m);
            filesystem::mode mode() const;
    };
    
    namespace handle {
        
        class source : public filehandle_source_sink {
            public:
                source();
                source(FILE* fh);
                source(char* cpath);
                source(char const* ccpath);
                source(std::string& spath);
                source(std::string const& cspath);
                source(filesystem::path const& ppath);
        };
        
        class sink : public filehandle_source_sink {
            public:
                sink();
                sink(FILE* fh);
                sink(char* cpath);
                sink(char const* ccpath);
                sink(std::string& spath);
                sink(std::string const& cspath);
                sink(filesystem::path const& ppath);
        };
        
    } /* namespace handle */
    
} /* namespace im */

#endif /// LIBIMREAD_FILEHANDLE_HH_
