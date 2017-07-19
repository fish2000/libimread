/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FILE_HH_
#define LIBIMREAD_FILE_HH_

#include <fcntl.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/seekable.hh>
#include <libimread/store.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class fd_source_sink : public byte_source, public byte_sink, public store::xattrmap {
        
        protected:
            static constexpr int kReadFlags         = O_RDWR | O_NONBLOCK;
            static constexpr int kWriteFlags        = O_RDWR | O_NONBLOCK | O_CREAT | O_EXCL | O_TRUNC;
            static constexpr int kWriteCreateMask   = 0644;
            
            static int open_read(char const* p);
            static int open_write(char const* p, int mask = kWriteCreateMask);
        
        public:
            static std::size_t max_descriptor_count();
            static std::size_t max_descriptor_count(std::size_t);
        
        public:
            fd_source_sink();
            fd_source_sink(int fd);
            
            virtual ~fd_source_sink();
            
            /// im::seekable methods
            virtual bool can_seek() const noexcept override;
            virtual bool can_store() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t pos) override;
            virtual std::size_t seek_relative(int delta) override;
            virtual std::size_t seek_end(int delta) override;
            
            /// im::byte_source and im::byte_sink methods
            virtual std::size_t read(byte* buffer, std::size_t n) override;
            virtual bytevec_t full_data() override;
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
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            virtual filesystem::file fh() const noexcept;
            virtual void fh(FILE* fh) noexcept;
            
            virtual bool exists() const noexcept;
            virtual int open(std::string const& spath, filesystem::mode fmode = filesystem::mode::READ);
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
            file_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(char* cpath,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(char const* ccpath,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(std::string& spath,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(std::string const& cspath,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(filesystem::path const& ppath,
                             filesystem::mode fmode = filesystem::mode::READ);
            
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            filesystem::mode mode(filesystem::mode m);
            filesystem::mode mode() const;
    };
    
    class FileSource final : public file_source_sink {
        
        public:
            FileSource();
            FileSource(char* cpath);
            FileSource(char const* ccpath);
            FileSource(std::string& spath);
            FileSource(std::string const& cspath);
            FileSource(filesystem::path const& ppath);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FileSource);
    };
    
    class FileSink final : public file_source_sink {
        
        public:
            FileSink();
            FileSink(char* cpath);
            FileSink(char const* ccpath);
            FileSink(std::string& spath);
            FileSink(std::string const& cspath);
            FileSink(filesystem::path const& ppath);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FileSink);
    };
    

}

#endif /// LIBIMREAD_FILE_HH_
