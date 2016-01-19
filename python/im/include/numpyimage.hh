
#include <memory>
#include <string>
#include <tuple>

#include <Python.h>
#include <structmember.h>

#include "structcode.hpp"
#include "numpy.hh"
#include <libimread/ext/errors/demangle.hh>

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
            std::tie(endianness, parsetokens, pairvec) = structcode::parse(code);
            
            if (!pairvec.size()) {
                PyErr_Format(PyExc_ValueError,
                    "Structcode %.200s parsed to zero-length", code);
                return NULL;
            }
            
            /// Make python list of tuples
            Py_ssize_t imax = static_cast<Py_ssize_t>(pairvec.size());
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
                  typename PythonImageType = NumpyImage>
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
                  typename PythonImageType = NumpyImage>
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
                        "bad arguments to image_init");
                    return -1;
            }
            
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<im::FileSource> input;
            std::unique_ptr<im::Image> output;
            
            if (!filename) {
                PyErr_SetString(PyExc_ValueError,
                    "No filename");
                return -1;
            }
            
            try {
                format = std::unique_ptr<im::ImageFormat>(
                    im::for_filename(filename));
            } catch (im::FormatNotFound& exc) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't find an I/O format for filename");
                return -1;
            }
            
            input = std::unique_ptr<im::FileSource>(
                new im::FileSource(filename));
            
            if (!input->exists()) {
                PyErr_SetString(PyExc_ValueError,
                    "Can't find an image file for filename");
                return -1;
            }
            
            output = std::unique_ptr<im::Image>(
                format->read(input.get(), &factory, opts));
            
            if (dtype) {
                self->dtype = dtype;
            } else {
                self->dtype = PyArray_DescrFromType(NPY_UINT8);
            }
            Py_INCREF(self->dtype);
            
            self->image = im::detail::dynamic_cast_unique<ImageType>(std::move(output));
            
            /// ALL IS WELL:
            return 0;
        }
        
        /// __repr__ implementation
        template <typename PythonImageType = NumpyImage>
        PyObject* repr(PythonImageType* im) {
            return PyString_FromFormat("<%s @ %p>",
                terminator::demangle(typeid(*im).name()), im);
        }
        
        template <typename PythonImageType = NumpyImage>
        int getbuffer(PyObject* self, Py_buffer* view, int flags) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            int out = pyim->image->populate_buffer(view,
                                                   (NPY_TYPES)pyim->dtype->type_num,
                                                   flags);
            return out;
        }
        
        template <typename PythonImageType = NumpyImage>
        void releasebuffer(PyObject* self, Py_buffer* view) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            pyim->image->release_buffer(view);
            PyBuffer_Release(view);
        }
        
        /// DEALLOCATE
        template <typename PythonImageType = NumpyImage>
        void dealloc(PythonImageType* self) {
            self->cleanup();
            self->ob_type->tp_free((PyObject*)self);
        }
        
        /// NumpyImage.datatype getter
        template <typename PythonImageType = NumpyImage>
        PyObject*    get_dtype(PythonImageType* self, void* closure) {
            return Py_BuildValue("O", self->dtype);
        }
        
        template <typename PythonImageType = NumpyImage>
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
