/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_

#include <cstdio>
#include <vector>
#include <string>
#include <Python.h>
#include <structmember.h>
#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace py {
    
    namespace buffer {
        
        using im::byte;
        
        class source : public im::byte_source {
            public:
                source(Py_buffer const& pb);
                virtual ~source();
                virtual std::size_t read(byte* buffer, std::size_t n);
                
                virtual bool can_seek() const noexcept;
                virtual std::size_t seek_absolute(std::size_t p);
                virtual std::size_t seek_relative(int delta);
                virtual std::size_t seek_end(int delta);
                
                operator Py_buffer() const { return view; }
                std::string str() const;
                
            private:
                Py_buffer view;
                std::size_t pos;
        };
        
        class sink : public im::byte_sink {
            public:
                sink(Py_buffer const& pb);
                virtual ~sink();
                
                virtual bool can_seek() const noexcept;
                virtual std::size_t seek_absolute(std::size_t p);
                virtual std::size_t seek_relative(int delta);
                virtual std::size_t seek_end(int delta);
                
                virtual std::size_t write(const void* buffer, std::size_t n);
                virtual void flush();
                virtual std::vector<byte> contents();
                
                operator Py_buffer() const { return view; }
                std::string str() const;
                
            private:
                Py_buffer view;
                std::size_t pos;
        };
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYBUFFER_HPP_