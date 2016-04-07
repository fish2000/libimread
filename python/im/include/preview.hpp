/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PREVIEW_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PREVIEW_HPP_

#include <string>
#include <vector>
#include <Python.h>
#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
// #include "hybrid.hh"

namespace py {
    
    namespace image {
        
        using filesystem::path;
        static const int sleeptime = 2000;
        
        void preview(path const& timage);
        
    }
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PREVIEW_HPP_