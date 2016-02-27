
#include <string>
#include <libimread/errors.hh>
#include "check.hh"
#include "gil.hh"
#include "pybuffer.hpp"
#include "options.hpp"

namespace py {
    
    namespace options {
        
        const char* get_blob(PyObject* data,
                             std::size_t& len) noexcept {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(data)) { return nullptr; }
                len = PyString_Size(data);
                return PyString_AS_STRING(data);
            #elif PY_MAJOR_VERSION >= 3
                len = PyBytes_Size(data);
                if (!PyBytes_Check(data)) { return nullptr; }
                return PyBytes_AsString(data);
            #endif /// PY_MAJOR_VERSION
        }
        
        const char* get_cstring(PyObject* stro) noexcept {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(stro)) { return nullptr; }
                return PyString_AS_STRING(stro);
            #elif PY_MAJOR_VERSION >= 3
                if (!PyUnicode_Check(stro)) { return nullptr; }
                return PyUnicode_AsUTF8(stro);
            #endif /// PY_MAJOR_VERSION
        }
        
        options_list parse_option_list(PyObject* list) {
            options_list out;
            if (!list) { return out; }
            if (!PySequence_Check(list)) { return out; }
            PyObject* sequence = PySequence_Fast(list, "Sequence expected");
            int idx = 0, len = PySequence_Fast_GET_SIZE(sequence);
            for (; idx < len; idx++) {
                PyObject* item = PySequence_Fast_GET_ITEM(sequence, idx);
                if (PyDict_Check(item)) {
                    try {
                        out.append(py::options::parse_options(item));
                    } catch (im::OptionsError& exc) {
                        Py_DECREF(sequence);
                        throw;
                    }
                } else if (PyTuple_Check(item) || PyList_Check(item) || PyAnySet_Check(item)) {
                    try {
                        out.append(py::options::parse_option_list(item));
                    } catch (im::OptionsError& exc) {
                        Py_DECREF(sequence);
                        throw;
                    }
                } else if (item == Py_None) {
                    out.append(options_list::null);
                } else if (PyBool_Check(item)) {
                    out.append(PyObject_IsTrue(item) == 1);
                } else if (PyLong_Check(item)) {
                    out.append(PyLong_AsLong(item));
                } else if (PyFloat_Check(item)) {
                    out.append(PyFloat_AS_DOUBLE(item));
            #if PY_MAJOR_VERSION < 3
                } else if (PyInt_Check(item)) {
                    out.append(static_cast<int>(PyInt_AS_LONG(item)));
            #elif PY_MAJOR_VERSION >= 3
                } else if (PyBytes_Check(item)) {
                    std::size_t len;
                    const char* blob = py::options::get_blob(item, len);
                    out.append(std::string(blob, len));
            #endif /// PY_MAJOR_VERSION
                } else if (PyMemoryView_Check(item)) {
                    Py_buffer* view = PyMemoryView_GET_BUFFER(item);
                    {
                        py::gil::release nogil;
                        py::buffer::source data = py::buffer::source(*view, false);
                        out.append(std::string(data.str()));
                    }
                } else if (NumpyImage_Check(item)) {
                    Py_DECREF(sequence);
                    imread_raise(OptionsError, "[ERROR]",
                        FF("Illegal NumpyImage data found in item %i (of %i)",
                            idx, len));
                } else {
                    const char* c = py::options::get_cstring(item);
                    if (!c) {
                        Py_DECREF(sequence);
                        imread_raise(OptionsError, "[ERROR]",
                            FF("Misparsed sequence item %i (of %i)",
                                idx, len));
                    }
                    out.append(std::string(c));
                }
            }
            Py_DECREF(sequence);
            return out;
        }
        
        options_map parse_options(PyObject* dict) {
            options_map out;
            if (!dict) { return out; }
            if (!PyDict_Check(dict)) { return out; }
            PyObject* key;
            PyObject* value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict, &pos, &key, &value)) {
                std::string k = py::options::get_cstring(key);
                if (PyDict_Check(value)) {
                    out.set(k, py::options::parse_options(value));
                } else if (PyTuple_Check(value) || PyList_Check(value) || PyAnySet_Check(value)) {
                    out.set(k, py::options::parse_option_list(value));
                } else if (value == Py_None) {
                    out.set(k, options_map::null);
                } else if (PyBool_Check(value)) {
                    out.set(k, PyObject_IsTrue(value) == 1);
                } else if (PyLong_Check(value)) {
                    out.set(k, PyLong_AsLong(value));
                } else if (PyFloat_Check(value)) {
                    out.set(k, PyFloat_AS_DOUBLE(value));
            #if PY_MAJOR_VERSION < 3
                } else if (PyInt_Check(value)) {
                    out.set(k, static_cast<int>(PyInt_AS_LONG(value)));
            #elif PY_MAJOR_VERSION >= 3
                } else if (PyBytes_Check(value)) {
                    std::size_t len;
                    const char* blob = py::options::get_blob(value, len);
                    out.set(k, std::string(blob, len));
            #endif /// PY_MAJOR_VERSION
                } else if (PyMemoryView_Check(value)) {
                    Py_buffer* view = PyMemoryView_GET_BUFFER(value);
                    {
                        py::gil::release nogil;
                        py::buffer::source data = py::buffer::source(*view, false);
                        out.set(k, std::string(data.str()));
                    }
                } else if (NumpyImage_Check(value)) {
                    imread_raise(OptionsError, "[ERROR]",
                        FF("Illegal NumpyImage data found at key %s (index %i)",
                            k.c_str(), pos));
                } else {
                    const char* c = py::options::get_cstring(value);
                    if (!c) {
                        imread_raise(OptionsError, "[ERROR]",
                            FF("Misparsed map value at key %s (index %i)",
                                k.c_str(), pos));
                    }
                    out.set(k, std::string(c));
                }
            }
            return out;
        }
        
    }
    
}