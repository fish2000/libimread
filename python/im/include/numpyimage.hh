
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_

#include <memory>
#include <string>
#include <tuple>

#include <Python.h>
#include <structmember.h>

#include "structcode.hpp"
#include "options.hpp"
#include "numpy.hh"
#include "gil.hh"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/memory.hh>
#include <libimread/hashing.hh>

namespace im {
    
    namespace detail {
        
        /// XXX: remind me why in fuck did I write this shit originally
        template <typename T, typename pT>
        std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT>&& source) {
            /// Force a dynamic_cast upon a unique_ptr via interim swap
            /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
            /// ... from http://stackoverflow.com/a/14777419/298171
            if (!source) { return std::unique_ptr<T>(); }
            
            /// Throws a std::bad_cast() if this doesn't work out
            T *destination = &dynamic_cast<T&>(*source.get());
            
            source.release();
            std::unique_ptr<T> out(destination);
            return out;
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
        using im::options_map;
        using im::HybridArray;
        using im::ArrayFactory;
        
        template <typename ImageType>
        struct PythonImageBase {
            PyObject_HEAD
            std::unique_ptr<ImageType> image;
            PyArray_Descr* dtype = nullptr;
            PyObject* readoptDict = nullptr;
            PyObject* writeoptDict = nullptr;
            
            options_map readopts() {
                return py::options::parse_options(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse_options(writeoptDict);
            }
            
            void cleanup() {
                image.release();
                Py_XDECREF(dtype);
                Py_XDECREF(readoptDict);
                Py_XDECREF(writeoptDict);
                dtype = nullptr;
                readoptDict = nullptr;
                writeoptDict = nullptr;
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
                self->dtype = nullptr;
                self->readoptDict = nullptr;
                self->writeoptDict = nullptr;
            }
            return reinterpret_cast<PyObject*>(self); /// all is well, return self
        }
        
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory>
        std::unique_ptr<ImageType> load(char const* source, options_map const& opts) {
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<im::FileSource> input;
            std::unique_ptr<im::Image> output;
            bool exists = false;
            options_map default_opts;
            
            try {
                py::gil::release nogil;
                format = std::unique_ptr<im::ImageFormat>(
                    im::for_filename(source));
            } catch (im::FormatNotFound& exc) {
                PyErr_Format(PyExc_ValueError,
                    "Can't find I/O format for file: %.200s", source);
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            {
                py::gil::release nogil;
                input = std::unique_ptr<im::FileSource>(
                    new im::FileSource(source));
                exists = input->exists();
            }
            
            if (!exists) {
                PyErr_Format(PyExc_ValueError,
                    "Can't find image file: %.200s", source);
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            {
                py::gil::release nogil;
                default_opts = format->get_options();
                output = std::unique_ptr<im::Image>(
                    format->read(input.get(), &factory, opts.update(default_opts)));
                return im::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
        }
        
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory>
        std::unique_ptr<ImageType> loadblob(char const* source, options_map const& opts) {
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<im::memory_source> input;
            std::unique_ptr<im::Image> output;
            options_map default_opts;
            
            try {
                py::gil::release nogil;
                input = std::unique_ptr<im::memory_source>(
                    new im::memory_source(reinterpret_cast<const byte*>(source),
                                          std::strlen(source)));
                format = std::unique_ptr<im::ImageFormat>(
                    im::for_source(input.get()));
                default_opts = format->get_options();
                output = std::unique_ptr<im::Image>(
                    format->read(input.get(), &factory, opts.update(default_opts)));
                return im::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            } catch (im::FormatNotFound& exc) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't find I/O format for blob source");
                return std::unique_ptr<ImageType>(nullptr);
            }
            
        }
        
        /// __init__ implementation
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int init(PyObject* self, PyObject* args, PyObject* kwargs) {
            static const std::unique_ptr<ImageType> unique_null_ptr = std::unique_ptr<ImageType>(nullptr);
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            PyObject* py_is_blob = NULL;
            PyObject* options = NULL;
            bool is_blob = false;
            char const* source = NULL;
            char const* keywords[] = { "source", "is_blob", "options", NULL };
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "s|O!O", const_cast<char**>(keywords),
                &source,
                &PyBool_Type, &py_is_blob,
                &options)) {
                    PyErr_SetString(PyExc_ValueError,
                        "Bad arguments to image_init");
                    return -1;
            } else {
                if (py_is_blob) {
                    is_blob = PyObject_IsTrue(py_is_blob);
                }
            }
            
            if (!source) {
                PyErr_SetString(PyExc_ValueError, "No filename or blob data");
                return -1;
            }
            
            try {
                if (is_blob) {
                    pyim->image = py::image::loadblob<ImageType, FactoryType>(
                        source, py::options::parse_options(options));
                } else {
                    pyim->image = py::image::load<ImageType, FactoryType>(
                        source, py::options::parse_options(options));
                }
            } catch (im::OptionsError& exc) {
                PyErr_SetString(PyExc_AttributeError, exc.what());
                return -1;
            }
            if (pyim->image == unique_null_ptr) {
                /// if this is true, PyErr has already been set
                return -1;
            }
            
            pyim->dtype = PyArray_DescrFromType(pyim->image->dtype());
            Py_INCREF(pyim->dtype);
            
            pyim->readoptDict = options ? options : PyDict_New();
            Py_INCREF(pyim->readoptDict);
            
            /// ALL IS WELL:
            return 0;
        }
        
        /// __repr__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* repr(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            char const* pytypename;
            {
                py::gil::release nogil;
                pytypename = terminator::nameof(pyim);
            }
            return PyString_FromFormat(
                "< %s @ %p >",
                pytypename, pyim);
        }
        
        /// __str__ implementaton -- return bytes of image
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* str(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            Py_ssize_t string_size;
            {
                py::gil::release nogil;
                string_size = pyim->image->size();
            }
            return PyString_FromStringAndSize(
                pyim->image->template rowp_as<char const>(0),
                string_size);
        }
        
        /// __hash__ implementation
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        long hash(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
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
        Py_ssize_t length(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            Py_ssize_t out;
            {
                py::gil::release nogil;
                out = pyim->image->size();
            }
            return out;
        }
        
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* atindex(PyObject* self, Py_ssize_t idx) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            Py_ssize_t siz;
            {
                py::gil::release nogil;
                siz = pyim->image->size();
            }
            if (siz <= idx || idx < 0) {
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
        void dealloc(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            pyim->cleanup();
            self->ob_type->tp_free(self);
        }
        
        /// NumpyImage.dtype getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_dtype(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->dtype);
        }
        
        /// NumpyImage.shape getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_shape(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image->dim(0));
                case 2:
                    return Py_BuildValue("(ii)",    image->dim(0),
                                                    image->dim(1));
                case 3:
                    return Py_BuildValue("(iii)",   image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2),
                                                    image->dim(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2),
                                                    image->dim(3),
                                                    image->dim(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        /// NumpyImage.strides getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_strides(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image->stride(0));
                case 2:
                    return Py_BuildValue("(ii)",    image->stride(0),
                                                    image->stride(1));
                case 3:
                    return Py_BuildValue("(iii)",   image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2),
                                                    image->stride(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2),
                                                    image->stride(3),
                                                    image->stride(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        /// NumpyImage.read_opts getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_read_opts(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->readoptDict);
        }
        
        /// NumpyImage.read_opts setter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int          set_read_opts(PyObject* self, PyObject* value, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            if (!value) { return 0; }
            Py_XDECREF(pyim->readoptDict);
            pyim->readoptDict = Py_BuildValue("O", value);
            return 0;
        }
        
        /// NumpyImage.write_opts getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_write_opts(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->writeoptDict);
        }
        
        /// NumpyImage.write_opts setter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int          set_write_opts(PyObject* self, PyObject* value, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            if (!value) { return 0; }
            Py_XDECREF(pyim->writeoptDict);
            pyim->writeoptDict = Py_BuildValue("O", value);
            return 0;
        }
        
    } /* namespace image */
        
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_
