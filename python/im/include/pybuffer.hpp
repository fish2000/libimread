/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_

#include <string>
#include <vector>
#include <Python.h>
#include <libimread/seekable.hh>

namespace py {
    
    namespace buffer {
        
        using im::byte;
        
        class source : public im::byte_source {
            public:
                source(Py_buffer const& pb);
                source(Py_buffer const& pb, bool r);
                virtual ~source();
                virtual std::size_t read(byte* buffer, std::size_t n);
                
                virtual bool can_seek() const noexcept;
                virtual std::size_t seek_absolute(std::size_t p);
                virtual std::size_t seek_relative(int delta);
                virtual std::size_t seek_end(int delta);
                virtual std::vector<byte> full_data();
                virtual std::size_t size() const;
                virtual void* readmap(std::size_t pageoffset = 0) const;
                
                operator Py_buffer() const { return view; }
                std::string str() const;
                
            private:
                Py_buffer view;
                std::size_t pos;
                bool release;
        };
        
        class sink : public im::byte_sink {
            public:
                sink(Py_buffer& pb);
                sink(Py_buffer& pb, bool r);
                virtual ~sink();
                
                virtual bool can_seek() const noexcept;
                virtual std::size_t seek_absolute(std::size_t p);
                virtual std::size_t seek_relative(int delta);
                virtual std::size_t seek_end(int delta);
                
                virtual std::size_t write(const void* buffer, std::size_t n);
                virtual void flush();
                virtual std::vector<byte> contents() const;
                
                operator Py_buffer() const { return view; }
                std::string str() const;
                
            private:
                Py_buffer view;
                std::size_t pos;
                bool release;
        };
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_