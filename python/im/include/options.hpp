
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_OPTIONS_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_OPTIONS_HH_

#include <cstddef>
#include <Python.h>
#include <libimread/options.hh>

namespace py {
    
    namespace options {
        
        using im::options_map;
        using im::options_list;
        
        const char* get_blob(PyObject* data, std::size_t& len) noexcept;
        const char* get_cstring(PyObject* stro) noexcept;
        options_list parse_option_list(PyObject* list);
        options_list parse_option_set(PyObject* set);
        options_map parse_options(PyObject* dict);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_OPTIONS_HH_