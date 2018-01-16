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
            memory_source(const byte*, const int);
            virtual ~memory_source();
            
            virtual std::size_t read(byte*, std::size_t) const;
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t);
            virtual std::size_t seek_relative(int);
            virtual std::size_t seek_end(int);
            virtual bytevec_t full_data() const;
            virtual std::size_t size() const;
            
            virtual void* readmap(std::size_t pageoffset = 0) const;
        
        private:
            const byte* data;
            const std::size_t length;
            mutable std::size_t pos;
    };
    
    class memory_sink : public byte_sink {
        public:
            memory_sink(byte*, std::size_t);
            memory_sink(std::size_t);
            virtual ~memory_sink();
            
            virtual bool can_seek() const noexcept;
            virtual std::size_t seek_absolute(std::size_t);
            virtual std::size_t seek_relative(int);
            virtual std::size_t seek_end(int);
            
            virtual std::size_t write(const void*, std::size_t);
            virtual void flush();
            
            virtual bytevec_t contents();
            
        private:
            byte* data;
            memory::buffer membuf;
            const std::size_t length;
            const bool allocated;
    };

}

#endif /// LIBIMREAD_MEMORY_HH_
