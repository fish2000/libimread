
#include "options.hh"
#include <libimread/errors.hh>

namespace py {
    
    namespace options {
        
        const char* get_blob(PyObject* data, std::size_t& len) {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(data)) { return 0; }
                len = PyString_Size(data);
                return PyString_AsString(data);
            #else
                len = PyBytes_Size(data);
                if (!PyBytes_Check(data)) { return 0; }
                return PyBytes_AsString(data);
            #endif
        }
        
        const char* get_cstring(PyObject* stro) {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(stro)) { return 0; }
                return PyString_AsString(stro);
            #else
                if (!PyUnicode_Check(stro)) { return 0; }
                return PyUnicode_AsUTF8(stro);
            #endif
        }
        
        options_map parse_options(PyObject* dict) {
            options_map out;
            if (!PyDict_Check(dict)) { return out; }
            PyObject* key;
            PyObject* value;
            Py_ssize_t pos = 0;
            
            while (PyDict_Next(dict, &pos, &key, &value)) {
                std::string k = py::options::get_cstring(key);
                if (PyLong_Check(value)) {
                    out[k] = int(PyLong_AsLong(value));
            #if PY_MAJOR_VERSION < 3
                } else if (PyInt_Check(value)) {
                    out[k] = int(PyInt_AS_LONG(value));
            #endif
                } else if (PyFloat_Check(value)) {
                    out[k] = PyFloat_AS_DOUBLE(value);
            #if PY_MAJOR_VERSION >= 3
                } else if (PyBytes_Check(value)) {
                    std::size_t len;
                    const char* blob = py::options::get_blob(value, len);
                    out[k] = std::string(blob, len);
            #endif
                } else {
                    const char* c = py::options::get_cstring(value);
                    if (!c) {
                        imread_raise(OptionsError,
                            "Type not understood while parsing option value:", k);
                    }
                    out[k] = std::string(c);
                }
            }
            return out;
        }
        
        
    }
    
}