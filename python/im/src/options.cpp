
#include <string>

#include "options.hpp"
#include "check.hh"
#include "detail.hpp"
#include "exceptions.hpp"
#include "gil.hpp"
#include "gil-io.hpp"
#include "pybuffer.hpp"
#include <libimread/errors.hh>

namespace py {
    
    namespace options {
        
        bool truth(PyObject* value) noexcept {
            if (!value) { return false; }
            return PyObject_IsTrue(value) == 1;
        }
        
        char const* get_blob(PyObject* data,
                             std::size_t& len) noexcept {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(data))  { return nullptr; }
                len = PyString_Size(data);
                return PyString_AS_STRING(data);
            #elif PY_MAJOR_VERSION >= 3
                len = PyBytes_Size(data);
                if (!PyBytes_Check(data))   { return nullptr; }
                return PyBytes_AsString(data);
            #endif /// PY_MAJOR_VERSION
        }
        
        char const* get_cstring(PyObject* stro) noexcept {
            #if PY_MAJOR_VERSION < 3
                if (!PyString_Check(stro))  { return nullptr; }
                return PyString_AS_STRING(stro);
            #elif PY_MAJOR_VERSION >= 3
                if (!PyUnicode_Check(stro)) { return nullptr; }
                return PyUnicode_AsUTF8(stro);
            #endif /// PY_MAJOR_VERSION
        }
        
        Json convert(PyObject* value) {
            if (PyDict_Check(value)) {
                return py::options::parse(value);
            } else if (PyTuple_Check(value) || PyList_Check(value)) {
                return py::options::parse_list(value);
            } else if (PyAnySet_Check(value)) {
                return py::options::parse_set(value);
            } else if (value == Py_None) {
                return options_map::null;
            } else if (PyBool_Check(value)) {
                return py::options::truth(value);
            } else if (PyLong_Check(value)) {
                return PyLong_AsLong(value);
            } else if (PyFloat_Check(value)) {
                return PyFloat_AS_DOUBLE(value);
        #if PY_MAJOR_VERSION < 3
            } else if (PyInt_Check(value)) {
                return static_cast<int>(PyInt_AS_LONG(value));
        #elif PY_MAJOR_VERSION >= 3
            } else if (PyBytes_Check(value)) {
                std::size_t len;
                char const* blob = py::options::get_blob(value, len);
                return std::string(blob, len);
        #endif /// PY_MAJOR_VERSION
            } else if (PyMemoryView_Check(value)) {
                Py_buffer* view = PyMemoryView_GET_BUFFER(value);
                {
                    py::gil::release nogil;
                    py::buffer::source data = py::buffer::source(*view, false);
                    return data.str();
                }
            } else if (PyFile_Check(value)) {
                {
                    py::gil::with iohandle(value);
                    auto fulldata = iohandle.source()->full_data();
                    return std::string((const char*)&fulldata[0], fulldata.size());
                }
            } else if (HybridImage_Check(value)         ||
                       BufferModel_Check(value)         ||
                       ImageModel_Check(value)          ||
                       ImageBufferModel_Check(value)) {
                return py::ValueError(
                    "Illegal data found",
                    options_map::undefined);
            } else if (PyObject_CheckBuffer(value)) {
                Py_buffer view;
                PyObject_GetBuffer(value, &view, PyBUF_SIMPLE);
                {
                    py::gil::release nogil;
                    py::buffer::source data = py::buffer::source(view);
                    return data.str();
                }
            }
            /// "else":
            char const* c = py::options::get_cstring(value);
            if (!c) {
                return py::KeyError(
                    "Misparsed map value",
                    options_map::undefined);
            }
            return c;
        }
        
        options_list parse_list(PyObject* list) {
            options_list out;
            if (!list) { return out; }
            if (!PySequence_Check(list)) { return out; }
            py::ref sequence = PySequence_Fast(list, "Sequence expected");
            int idx = 0,
                len = PySequence_Fast_GET_SIZE(sequence.get());
            for (; idx < len; idx++) {
                PyObject* item = PySequence_Fast_GET_ITEM(sequence.get(), idx);
                Json ison(Json::null);
                ison = py::options::convert(item); /// might PyErr here
                out.append(ison);
            }
            return out;
        }
        
        options_list parse_set(PyObject* set) {
            options_list out;
            if (!set) { return out; }
            if (!PyAnySet_Check(set)) { return out; }
            py::ref iterator = PyObject_GetIter(set);
            if (iterator.empty()) {
                return py::ValueError(
                    "Set object not iterable",
                    options_list::undefined);
            }
            while (py::ref item = PyIter_Next(iterator)) {
                Json ison(Json::null);
                ison = py::options::convert(item); /// might PyErr here
                out.append(ison);
            }
            if (PyErr_Occurred()) {
                return py::IOError(
                    "Error occurred while iterating set",
                    options_list::undefined);
            }
            return out;
        }
        
        options_map parse(PyObject* dict) {
            options_map out;
            if (!dict) { return out; }
            if (!PyDict_Check(dict)) { return out; }
            PyObject* key;
            PyObject* value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict, &pos, &key, &value)) {
                std::string k = py::options::get_cstring(key);
                Json v(Json::null);
                v = py::options::convert(value); /// might PyErr here
                out.set(k, v);
            }
            return out;
        }
        
        PyObject* revert(Json const& value) {
            switch (value.type()) {
                case Type::JSNULL:
                    return py::None();
                case Type::BOOLEAN:
                    return py::boolean((bool)value);
                case Type::NUMBER:
                    return value.is_integer() ? py::convert((long)value) :
                                                py::convert((long double)value);
                case Type::STRING:
                    return py::string((std::string)value);
                case Type::ARRAY: {
                    int max = value.size();
                    PyObject* list = PyList_New(max);
                    for (int idx = 0; idx < max; ++idx) {
                        Json subvalue(value[idx]);
                        PyList_SET_ITEM(list, idx,
                            py::options::revert(subvalue));
                    }
                    return list;
                }
                case Type::OBJECT: {
                    auto const& keys = value.keys();
                    PyObject* dict = PyDict_New();
                    int idx = 0,
                        max = keys.size();
                    for (auto it = keys.begin();
                         it != keys.end() && idx < max;
                         ++it) { std::string const& key{*it};
                                 if (key.size() > 0) {
                                     Json subvalue(value[key]);
                                     py::detail::setitemstring(dict, key,
                                         py::options::revert(subvalue));
                                 } ++idx; }
                    return dict;
                }
                case Type::POINTER:
                    return py::string("<POINTER>");
                case Type::SCHEMA:
                    return py::string("<SCHEMA>");
                default:
                    return py::string("<unknown>");
            }
        }
        
        PyObject* dump(PyObject* self, PyObject* args, PyObject* kwargs,
                       options_map const& opts) {
            PyObject* py_overwrite = nullptr;
            PyObject* py_tempfile = nullptr;
            char const* keywords[] = { "destination", "overwrite", "tempfile", nullptr };
            char const* destination = nullptr;
            bool overwrite = false;
            bool tempfile = false;
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "|sOO", const_cast<char**>(keywords),
                &destination,
                &py_overwrite,
                &py_tempfile))
                    { return nullptr; }
            
            if (!py_tempfile && !destination) {
                return py::AttributeError("Must specify either destination path or tempfile=True");
            }
            overwrite = py::options::truth(py_overwrite);
            tempfile  = py::options::truth(py_tempfile);
            
            try {
                py::gil::release nogil;
                if (tempfile) {
                    std::string dest(opts.dumptmp());
                    destination = dest.c_str();
                } else {
                    opts.dump(destination, overwrite);
                }
            } catch (im::JSONIOError& exc) {
                return py::IOError(exc.what());
            }
            
            return py::string(destination);
        }
        
    } /* namespace options */
    
} /* namespace py */