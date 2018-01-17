/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
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
#include <libimread/buffered_io.hh>
#include <libimread/store.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class fd_source_sink : public byte_source, public byte_sink, public store::xattrmap {
        
        protected:
            static constexpr int kReadFlags         = O_RDWR | O_NONBLOCK | O_CLOEXEC;
            static constexpr int kWriteFlags        = O_RDWR | O_NONBLOCK | O_CLOEXEC | O_CREAT | O_EXCL | O_TRUNC;
            static constexpr int kFIFOReadFlags     = O_RDONLY | O_CLOEXEC;
            static constexpr int kFIFOWriteFlags    = O_WRONLY | O_CLOEXEC;
            static constexpr int kWriteCreateMask   = 0644;
            
        protected:
            static int open_read(char const*);
            static int open_write(char const*, int mask = kWriteCreateMask);
            static int fifo_open_read(char const*);
            static int fifo_open_write(char const*);
            static bool check_descriptor(int) noexcept;
        
        public:
            static std::size_t max_descriptor_count();
            static std::size_t max_descriptor_count(std::size_t);
        
        public:
            fd_source_sink() noexcept = default;
            fd_source_sink(int) noexcept;
            
            fd_source_sink(fd_source_sink const&);
            fd_source_sink(fd_source_sink&&) noexcept;
            
            virtual ~fd_source_sink();
            
        public:
            /// im::seekable methods
            virtual bool can_seek() const noexcept override;
            virtual bool can_store() const noexcept override;
            virtual std::size_t seek_absolute(std::size_t) override;
            virtual std::size_t seek_relative(int) override;
            virtual std::size_t seek_end(int) override;
            
        public:
            /// im::byte_source and im::byte_sink methods
            virtual std::size_t read(byte*, std::size_t) const override;
            virtual bytevec_t full_data() const override;
            virtual std::size_t size() const override;
            virtual std::size_t write(const void* buffer, std::size_t n) override;
            virtual std::size_t write(bytevec_t const&) override;
            virtual std::size_t write(bytevec_t&&) override;
            virtual detail::stat_t stat() const;
            virtual void flush() override;
            
        public:
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
            virtual filesystem::file fh() const noexcept;
            virtual void fh(FILE*) noexcept;
            
        public:
            virtual bool exists() const noexcept;
            virtual bool is_open() const noexcept;
            virtual int open(std::string const&, filesystem::mode fmode = filesystem::mode::READ);
            virtual int close();
            
        protected:
            int descriptor{ -1 };
            mutable detail::mapped_t mapped;
    };
    
    class file_source_sink : public fd_source_sink {
        
        public:
            file_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(char*,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(char const*,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(std::string&,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(std::string const&,
                             filesystem::mode fmode = filesystem::mode::READ);
            file_source_sink(filesystem::path const&,
                             filesystem::mode fmode = filesystem::mode::READ);
            
        public:
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            filesystem::mode mode(filesystem::mode);
            filesystem::mode mode() const;
        
        protected:
            filesystem::path pth;
            filesystem::mode md;
    };
    
    class fifo_source_sink : public fd_source_sink {
        
        public:
            fifo_source_sink(filesystem::mode fmode = filesystem::mode::READ);
            fifo_source_sink(char*,
                             filesystem::mode fmode = filesystem::mode::READ);
            fifo_source_sink(char const*,
                             filesystem::mode fmode = filesystem::mode::READ);
            fifo_source_sink(std::string&,
                             filesystem::mode fmode = filesystem::mode::READ);
            fifo_source_sink(std::string const&,
                             filesystem::mode fmode = filesystem::mode::READ);
            fifo_source_sink(filesystem::path const&,
                             filesystem::mode fmode = filesystem::mode::READ);
            
        public:
            filesystem::path const& path() const;
            virtual bool exists() const noexcept override;
            virtual int open(std::string const&, filesystem::mode fmode = filesystem::mode::READ) override;
            filesystem::mode mode(filesystem::mode);
            filesystem::mode mode() const;
        
        protected:
            filesystem::path pth;
            filesystem::mode md;
    };
    
    class FileSource final : public file_source_sink {
        
        public:
            FileSource();
            FileSource(char*);
            FileSource(char const*);
            FileSource(std::string&);
            FileSource(std::string const&);
            FileSource(filesystem::path const&);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FileSource);
    };
    
    class FileSink : public file_source_sink {
        
        public:
            FileSink();
            FileSink(char*);
            FileSink(char const*);
            FileSink(std::string&);
            FileSink(std::string const&);
            FileSink(filesystem::path const&);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FileSink);
    };
    
    class FIFOSource final : public fifo_source_sink {
        
        public:
            FIFOSource();
            FIFOSource(char*);
            FIFOSource(char const*);
            FIFOSource(std::string&);
            FIFOSource(std::string const&);
            FIFOSource(filesystem::path const&);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FIFOSource);
    };
    
    class FIFOSink : public fifo_source_sink {
        
        public:
            FIFOSink();
            FIFOSink(char*);
            FIFOSink(char const*);
            FIFOSink(std::string&);
            FIFOSink(std::string const&);
            FIFOSink(filesystem::path const&);
            
        public:
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(FIFOSink);
    };
    
    // using BufferedFileSink = buffered_sink<FileSink>;
    // using BufferedFIFOSink = buffered_sink<FIFOSink>;
    
    class BufferedFileSink final : public buffered_sink<FileSink> {
        
        public:
            using BufferedSink = buffered_sink<FileSink>;
            using BufferedSink::get_primary;
        
        public:
            BufferedFileSink();
            BufferedFileSink(char*);
            BufferedFileSink(char const*);
            BufferedFileSink(std::string&);
            BufferedFileSink(std::string const&);
            BufferedFileSink(filesystem::path const&);
        
        public:
            ~BufferedFileSink();
    };
    
    class BufferedFIFOSink final : public buffered_sink<FIFOSink> {
    
        public:
            using BufferedSink = buffered_sink<FIFOSink>;
            using BufferedSink::get_primary;
        
        public:
            BufferedFIFOSink();
            BufferedFIFOSink(char*);
            BufferedFIFOSink(char const*);
            BufferedFIFOSink(std::string&);
            BufferedFIFOSink(std::string const&);
            BufferedFIFOSink(filesystem::path const&);
        
        public:
            ~BufferedFIFOSink();
    };
    
}

#endif /// LIBIMREAD_FILE_HH_
