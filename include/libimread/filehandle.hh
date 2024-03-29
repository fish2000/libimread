/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FILEHANDLE_HH_
#define LIBIMREAD_FILEHANDLE_HH_

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/seekable.hh>
#include <libimread/store.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class handle_source_sink : public byte_source, public byte_sink, public store::xattrmap {
        
        public:
            handle_source_sink();
            handle_source_sink(FILE*);
            
            virtual ~handle_source_sink();
            
        public:
            /// im::seekable methods
            virtual bool can_seek() const noexcept override;
            virtual bool can_store() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t pos) override;
            virtual std::size_t seek_relative(int delta) override;
            virtual std::size_t seek_end(int delta) override;
            
        public:
            /// im::byte_source and im::byte_sink methods
            virtual std::size_t read(byte*, std::size_t) const override;
            virtual bytevec_t full_data() const override;
            virtual std::size_t size() const override;
            virtual std::size_t write(const void*, std::size_t) override;
            virtual std::size_t write(bytevec_t const&) override;
            virtual std::size_t write(bytevec_t&&) override;
            virtual detail::stat_t stat() const;
            virtual void flush() override;
            
            virtual void* readmap(std::size_t pageoffset = 0) const override;
            
        public:
            /// Filesystem extended attribute (“xattr”) access
            virtual std::string xattr(std::string const&) const override;
            virtual std::string xattr(std::string const&, std::string const&) const override;
            virtual int xattrcount() const override;
            virtual filesystem::detail::stringvec_t xattrs() const override;
            
        public:
            virtual int fd() const noexcept;
            virtual void fd(int) noexcept;
            virtual FILE* fh() const noexcept;
            virtual void fh(FILE*) noexcept;
            
        public:
            virtual bool exists() const noexcept;
            virtual FILE* open(char*, filesystem::mode fmode = filesystem::mode::READ);
            virtual FILE* close();
            
        protected:
            mutable FILE* handle = nullptr;
            mutable detail::mapped_t mapped;
            bool external = false;
    };
    
    class filehandle_source_sink : public handle_source_sink {
        
        public:
            filehandle_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(FILE*, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(char*, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(char const*, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(std::string&, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(std::string const&, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(filesystem::path const&, filesystem::mode fmode = filesystem::mode::READ);
            
        public:
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            filesystem::mode mode(filesystem::mode);
            filesystem::mode mode() const;
            
        protected:
            filesystem::path pth;
            filesystem::mode md;
    };
    
    namespace handle {
        
        class source : public filehandle_source_sink {
            
            public:
                DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(source);
            
            public:
                source();
                source(FILE*);
                source(char*);
                source(char const*);
                source(std::string&);
                source(std::string const&);
                source(filesystem::path const&);
        };
        
        class sink : public filehandle_source_sink {
            
            public:
                DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(sink);
            
            public:
                sink();
                sink(FILE*);
                sink(char*);
                sink(char const*);
                sink(std::string&);
                sink(std::string const&);
                sink(filesystem::path const&);
        };
        
    } /* namespace handle */
    
} /* namespace im */

#endif /// LIBIMREAD_FILEHANDLE_HH_
