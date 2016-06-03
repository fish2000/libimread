/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_MEMORY_HH_
#define LIBIMREAD_MEMORY_HH_

#include <vector>
#include <libimread/libimread.hpp>
#include <libimread/ext/memory/fmemopen.hh>
#include <libimread/seekable.hh>

namespace im {
    
    class memory_source : public byte_source {
        public:
            memory_source(const byte* c, const int len);
            virtual ~memory_source();
            
            virtual std::size_t read(byte* buffer, std::size_t n);
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t p);
            virtual std::size_t seek_relative(int delta);
            virtual std::size_t seek_end(int delta);
            virtual std::vector<byte> full_data();
            virtual std::size_t size();
            
            virtual void* readmap(std::size_t pageoffset = 0);
        
        private:
            const byte* data;
            const std::size_t length;
            std::size_t pos;
    };
    
    class memory_sink : public byte_sink {
        public:
            memory_sink(byte* c, std::size_t len);
            memory_sink(std::size_t len);
            virtual ~memory_sink();
            
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t pos);
            virtual std::size_t seek_relative(int delta);
            virtual std::size_t seek_end(int delta);
            
            virtual std::size_t write(const void* buffer, std::size_t n);
            virtual std::size_t write(std::vector<byte> const& bv);
            virtual void flush();
            
            virtual std::vector<byte> contents();
            
        private:
            byte* data;
            memory::buffer membuf;
            const std::size_t length;
            const bool allocated;
    };

}

#endif /// LIBIMREAD_MEMORY_HH_
