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
#include <libimread/ext/filesystem/path.h>
#include <libimread/seekable.hh>

namespace im {
    
    namespace detail {
        using stat_t = struct stat;
        using mapped_t = std::unique_ptr<void, std::function<void(void*)>>;
    }
    
    class handle_source_sink : public byte_source, public byte_sink {
        
        public:
            handle_source_sink();
            handle_source_sink(FILE* fh);
            
            virtual ~handle_source_sink();
            
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t pos);
            virtual std::size_t seek_relative(int delta);
            virtual std::size_t seek_end(int delta);
            
            virtual std::size_t read(byte* buffer, std::size_t n);
            virtual std::vector<byte> full_data();
            virtual std::size_t size();
            virtual std::size_t write(const void* buffer, std::size_t n);
            virtual std::size_t write(std::vector<byte> const& bv);
            virtual detail::stat_t stat() const;
            virtual void flush();
            
            virtual void* readmap(std::size_t pageoffset = 0);
            
            virtual int fd() const noexcept;
            virtual void fd(int fd) noexcept;
            virtual FILE* fh() const noexcept;
            virtual void fh(FILE* fh) noexcept;
            
            virtual bool exists() const noexcept;
            virtual FILE* open(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            virtual FILE* close();
            
        private:
            FILE* handle = nullptr;
            detail::mapped_t mapped;
            bool external = false;
    };
    
    class filehandle_source_sink : public handle_source_sink {
        private:
            filesystem::path pth;
            filesystem::mode md;
        
        public:
            filehandle_source_sink(filesystem::mode fmode = filesystem::mode::READ)
                :handle_source_sink(), md(fmode)
                {}
            
            filehandle_source_sink(FILE* fh, filesystem::mode fmode = filesystem::mode::READ);
            filehandle_source_sink(char* cpath, filesystem::mode fmode = filesystem::mode::READ);
            
            filehandle_source_sink(char const* ccpath,
                                   filesystem::mode fmode = filesystem::mode::READ)
                :filehandle_source_sink(const_cast<char*>(ccpath), fmode)
                {}
            
            filehandle_source_sink(std::string& spath,
                                   filesystem::mode fmode = filesystem::mode::READ)
                :filehandle_source_sink(spath.c_str(), fmode)
                {}
            
            filehandle_source_sink(std::string const& cspath,
                                   filesystem::mode fmode = filesystem::mode::READ)
                :filehandle_source_sink(cspath.c_str(), fmode)
                {}
            
            filehandle_source_sink(filesystem::path const& ppath,
                                   filesystem::mode fmode = filesystem::mode::READ)
                :filehandle_source_sink(ppath.c_str(), fmode)
                {}
            
            filesystem::path const& path() const { return pth; }
            virtual bool exists() const noexcept override;
            
            void mode(filesystem::mode m) { md = m; }
            filesystem::mode mode() const { return md; }
    };
    
    namespace handle {
        
        class source : public filehandle_source_sink {
            public:
                source()
                    :filehandle_source_sink()
                    {}
                source(FILE* fh)
                    :filehandle_source_sink(fh)
                    {}
                source(char* cpath)
                    :filehandle_source_sink(cpath)
                    {}
                source(char const* ccpath)
                    :filehandle_source_sink(ccpath)
                    {}
                source(std::string& spath)
                    :filehandle_source_sink(spath)
                    {}
                source(std::string const& cspath)
                    :filehandle_source_sink(cspath)
                    {}
                source(filesystem::path const& ppath)
                    :filehandle_source_sink(ppath)
                    {}
        };
        
        class sink : public filehandle_source_sink {
            public:
                sink()
                    :filehandle_source_sink(filesystem::mode::WRITE)
                    {}
                sink(FILE* fh)
                    :filehandle_source_sink(fh, filesystem::mode::WRITE)
                    {}
                sink(char* cpath)
                    :filehandle_source_sink(cpath, filesystem::mode::WRITE)
                    {}
                sink(char const* ccpath)
                    :filehandle_source_sink(ccpath, filesystem::mode::WRITE)
                    {}
                sink(std::string& spath)
                    :filehandle_source_sink(spath, filesystem::mode::WRITE)
                    {}
                sink(std::string const& cspath)
                    :filehandle_source_sink(cspath, filesystem::mode::WRITE)
                    {}
                sink(filesystem::path const& ppath)
                    :filehandle_source_sink(ppath, filesystem::mode::WRITE)
                    {}
        };
        
        
    }
    

}

#endif /// LIBIMREAD_FILEHANDLE_HH_
