
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_

#include <memory>
#include <string>
#include <iostream>
#include <Python.h>
#include <structmember.h>

#include "check.hh"
#include "gil.hh"
#include "detail.hh"
#include "options.hpp"
#include "pybuffer.hpp"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/memory.hh>
#include <libimread/hashing.hh>

namespace py {
    
    namespace opts {
        
        using im::byte;
        using im::options_map;
        using im::options_list;
        //using Json;
        
        using filesystem::path;
        using filesystem::NamedTemporaryFile;
        
        template <typename OptionsType>
        struct PythonOptionsBase {
            PyObject_HEAD
            OptionsType* data;
            PyObject* source;
            
            void cleanup() {
                data->reset();
                Py_CLEAR(source);
            }
        };
        
        using OptionsMap  = PythonOptionsBase<options_map>;
        using OptionsList = PythonOptionsBase<options_list>;
        using MappingBase = PythonOptionsBase<Json>;
        
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
            PythonOptionsType* self = reinterpret_cast<PythonOptionsType*>(type->tp_alloc(type, 0));
            if (self != NULL) {
                self->data = new OptionsType(OptionsType::null);
                self->source = nullptr;
            }
            return reinterpret_cast<PyObject*>(self); /// all is well, return self
        }
        
        
        /// __init__ implementation
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        int init(PyObject* self, PyObject* args, PyObject* kwargs) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            // PyObject* py_is_blob = NULL;
            // PyObject* options = NULL;
            // Py_buffer view;
            // char const* keywords[] = { "source", "is_blob", "options", NULL };
            // bool is_blob = false;
            //
            // if (!PyArg_ParseTupleAndKeywords(
            //     args, kwargs, "s*|OO", const_cast<char**>(keywords),
            //     &view,                      /// "view", buffer with file path or image data
            //     &py_is_blob,                /// "is_blob", Python boolean specifying blobbiness
            //     &options))                  /// "options", read-options dict
            // {
            //     return -1;
            // }
            //
        }
        
        
        /// __repr__ implementation
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        PyObject* repr(PyObject* self) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            char const* pytypename;
            {
                py::gil::release nogil;
                pytypename = terminator::nameof(pyopts);
            }
            return PyString_FromFormat(
                "< %s @ %p >",
                pytypename, pyopts);
        }
        
        /// __str__ implementaton
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* str(PyObject* self) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            Py_ssize_t string_size;
            std::string out;
            {
                py::gil::release nogil;
                out = pyopts->data->format();
                string_size = out.size();
            }
            return PyString_FromStringAndSize(
                out.c_str(), string_size);
        }
        
        /// __hash__ implementation
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        long hash(PyObject* self) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            long out;
            {
                py::gil::release nogil;
                out = static_cast<long>(pyopts->data->hash());
            }
            return out;
        }
        
        /// __len__ implementation
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        Py_ssize_t length(PyObject* self) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            Py_ssize_t out;
            try {
                py::gil::release nogil;
                out = static_cast<Py_ssize_t>(pyopts->data->size());
            } except (im::JSONUseError& exc) {
                out = 0;
            }
            return out;
        }
        
        /// DEALLOCATE
        template <typename OptionsType = options_map,
                  typename PythonOptionsType = PythonOptionsBase<OptionsType>>
        void dealloc(PyObject* self) {
            PythonOptionsType* pyopts = reinterpret_cast<PythonOptionsType*>(self);
            pyopts->cleanup();
            delete pyopts->data;
            self->ob_type->tp_free(self);
        }
        
    } /* namespace opts */

} /* namespace py */