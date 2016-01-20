
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_STRUCTCODE_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_STRUCTCODE_HPP_

#include <memory>
#include <string>
#include <tuple>

#include <Python.h>
#include <structmember.h>

#include "structcode.hpp"
#include "numpy.hh"
#include "gil.hh"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/hashing.hh>

namespace im {
    
    namespace detail {
        
        /// XXX: remind me why in fuck did I write this shit originally
        template <typename T, typename pT>
        std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT>&& src) {
            /// Force a dynamic_cast upon a unique_ptr via interim swap
            /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
            /// ... from http://stackoverflow.com/a/14777419/298171
            if (!src) { return std::unique_ptr<T>(); }
            
            /// Throws a std::bad_cast() if this doesn't work out
            T *dst = &dynamic_cast<T&>(*src.get());
            
            src.release();
            std::unique_ptr<T> ret(dst);
            return ret;
        }
        
        PyObject* structcode_to_dtype(char const* code) {
            using structcode::stringvec_t;
            using structcode::structcode_t;
            using structcode::parse_result_t;
            
            std::string endianness;
            stringvec_t parsetokens;
            structcode_t pairvec;
            Py_ssize_t imax = 0;
            
            {
                py::gil::release nogil;
                std::tie(endianness, parsetokens, pairvec) = structcode::parse(code);
                imax = static_cast<Py_ssize_t>(pairvec.size());
            }
            
            if (!bool(imax)) {
                PyErr_Format(PyExc_ValueError,
                    "Structcode %.200s parsed to zero-length", code);
                return NULL;
            }
            
            /// Make python list of tuples
            PyObject* tuple = PyTuple_New(imax);
            for (Py_ssize_t idx = 0; idx < imax; idx++) {
                PyTuple_SET_ITEM(tuple, idx, PyTuple_Pack(2,
                    PyString_FromString(pairvec[idx].first.c_str()),
                    PyString_FromString((endianness + pairvec[idx].second).c_str())));
            }
            
            return tuple;
        }
        
    }
    
}

namespace py {
    
    namespace image {
        
        using im::byte;
        using im::HybridArray;
        using im::ArrayFactory;
        
        template <typename ImageType>
        struct PythonImageBase {
            PyObject_HEAD
            std::unique_ptr<ImageType> image;
            PyArray_Descr* dtype = nullptr;
            
            void cleanup() {
                image.release();
                Py_XDECREF(dtype);
                dtype = nullptr;
            }
            
            ~PythonImageBase() { cleanup(); }
        };
        
        using NumpyImage = PythonImageBase<HybridArray>;
        
        /// ALLOCATE / __new__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
            PythonImageType* self = reinterpret_cast<PythonImageType*>(type->tp_alloc(type, 0));
            /// initialize with defaults
            if (self != NULL) {
                self->image = std::unique_ptr<ImageType>(nullptr);
                self->dtype = NULL;
            }
            return reinterpret_cast<PyObject*>(self); /// all is well, return self
        }
        
        /// __init__ implementation
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int init(PythonImageType* self, PyObject* args, PyObject* kwargs) {
            PyArray_Descr* dtype = NULL;
            char const* filename = NULL;
            char const* keywords[] = { "file", "dtype", NULL };
            static const im::options_map opts; /// not currently used when reading
        
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "s|O&", const_cast<char**>(keywords),
                &filename,
                PyArray_DescrConverter, &dtype)) {
                    PyErr_SetString(PyExc_ValueError,
                        "Bad arguments to image_init");
                    return -1;
            }
            
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<im::FileSource> input;
            std::unique_ptr<im::Image> output;
            bool exists = false;
            
            if (!filename) {
                PyErr_SetString(PyExc_ValueError,
                    "No filename");
                return -1;
            }
            
            try {
                py::gil::release nogil;
                format = std::unique_ptr<im::ImageFormat>(
                    im::for_filename(filename));
            } catch (im::FormatNotFound& exc) {
                PyErr_Format(PyExc_ValueError,
                    "Can't find I/O format for file: %.200s",
                    filename);
                return -1;
            }
            
            {
                py::gil::release nogil;
                input = std::unique_ptr<im::FileSource>(
                    new im::FileSource(filename));
                exists = input->exists();
            }
            
            if (!exists) {
                PyErr_Format(PyExc_ValueError,
                    "Can't find image file: %.200s",
                    filename);
                return -1;
            }
            
            {
                py::gil::release nogil;
                output = std::unique_ptr<im::Image>(
                    format->read(input.get(), &factory, opts));
                self->image = im::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
            
            if (dtype) {
                self->dtype = dtype;
            } else {
                self->dtype = PyArray_DescrFromType(NPY_UINT8);
            }
            Py_INCREF(self->dtype);
            
            /// ALL IS WELL:
            return 0;
        }
        
        /// __repr__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* repr(PythonImageType* pyim) {
            decltype(terminator::nameof<*pyim>) pytypename;
            {
                py::gil::release nogil;
                pytypename = terminator::nameof(*pyim);
            }
            return PyString_FromFormat(
                "< %s @ %p >",
                pytypename, pyim);
        }
        
        /// __str__ implementaton -- return bytes of image
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* str(PythonImageType* pyim) {
            Py_ssize_t string_size;
            {
                py::gil::release nogil;
                string_size = pyim->image->size();
            }
            return PyString_FromStringAndSize(
                pyim->image->template rowp_as<char const*>(0),
                string_size);
        }
        
        /// __hash__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        long hash(PythonImageType* pyim) {
            long out;
            {
                py::gil::release nogil;
                auto bithash = blockhash::blockhash_quick(*pyim->image);
                out = static_cast<long>(bithash.to_ulong());
            }
            return out;
        }
        
        /// __len__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        Py_ssize_t length(PythonImageType* pyim) {
            Py_ssize_t out;
            {
                py::gil::release nogil;
                out = pyim->image->size();
            }
            return out;
        }
        
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* atindex(PythonImageType* pyim, Py_ssize_t idx) {
            if (pyim->image->size() <= idx) {
                PyErr_SetString(PyExc_IndexError,
                    "index out of range");
                return NULL;
            }
            int tc = static_cast<int>(pyim->dtype->type_num);
            std::size_t nidx = static_cast<std::size_t>(idx);
            switch (tc) {
                case NPY_FLOAT: {
                    float op = pyim->image->template rowp_as<float>(0)[nidx];
                    return Py_BuildValue("f", op);
                }
                break;
                case NPY_DOUBLE:
                case NPY_LONGDOUBLE: {
                    double op = pyim->image->template rowp_as<double>(0)[nidx];
                    return Py_BuildValue("d", op);
                }
                break;
                case NPY_USHORT:
                case NPY_UBYTE: {
                    byte op = pyim->image->template rowp_as<byte>(0)[nidx];
                    return Py_BuildValue("B", op);
                }
                break;
                case NPY_UINT: {
                    uint32_t op = pyim->image->template rowp_as<uint32_t>(0)[nidx];
                    return Py_BuildValue("I", op);
                }
                break;
                case NPY_ULONG:
                case NPY_ULONGLONG: {
                    uint64_t op = pyim->image->template rowp_as<uint64_t>(0)[nidx];
                    return Py_BuildValue("Q", op);
                }
                break;
            }
            uint32_t op = pyim->image->template rowp_as<uint32_t>(0)[nidx];
            return Py_BuildValue("I", op);
        }
        
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int getbuffer(PyObject* self, Py_buffer* view, int flags) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            int out;
            {
                py::gil::release nogil;
                out = pyim->image->populate_buffer(view,
                                                   (NPY_TYPES)pyim->dtype->type_num,
                                                   flags);
            }
            return out;
        }
        
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        void releasebuffer(PyObject* self, Py_buffer* view) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            {
                py::gil::release nogil;
                pyim->image->release_buffer(view);
            }
            PyBuffer_Release(view);
        }
        
        /// DEALLOCATE
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        void dealloc(PythonImageType* self) {
            self->cleanup();
            self->ob_type->tp_free((PyObject*)self);
        }
        
        /// NumpyImage.datatype getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_dtype(PythonImageType* self, void* closure) {
            return Py_BuildValue("O", self->dtype);
        }
        
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_shape(PythonImageType* self, void* closure) {
            switch (self->image->ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     self->image->dim(0));
                case 2:
                    return Py_BuildValue("(ii)",    self->image->dim(0),
                                                    self->image->dim(1));
                case 3:
                    return Py_BuildValue("(iii)",   self->image->dim(0),
                                                    self->image->dim(1),
                                                    self->image->dim(2));
                case 4:
                    return Py_BuildValue("(iiii)",  self->image->dim(0),
                                                    self->image->dim(1),
                                                    self->image->dim(2),
                                                    self->image->dim(3));
                case 5:
                    return Py_BuildValue("(iiiii)", self->image->dim(0),
                                                    self->image->dim(1),
                                                    self->image->dim(2),
                                                    self->image->dim(3),
                                                    self->image->dim(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
    } /* namespace image */
        
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_STRUCTCODE_HPP_
