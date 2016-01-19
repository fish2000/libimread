
#include <memory>
#include <string>
#include <tuple>

#include <Python.h>
#include <structmember.h>

#include "structcode.hpp"
#include "numpy.hh"

#ifndef SENTINEL
#define SENTINEL {NULL}
#endif

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif


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

using ImagePtr = std::unique_ptr<im::HybridArray>;

struct NumpyImage {
    PyObject_HEAD
    std::unique_ptr<im::HybridArray> image;
    PyArray_Descr* dtype = nullptr;
    
    void cleanup() {
        image.release();
        Py_XDECREF(dtype);
        dtype = nullptr;
    }
    
    ~NumpyImage() { cleanup(); }
};

namespace {
    
    /// ALLOCATE / __new__ implementation
    PyObject* NumpyImage_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
        NumpyImage* self;
        self = reinterpret_cast<NumpyImage*>(type->tp_alloc(type, 0));
        /// initialize with defaults
        if (self != NULL) {
            self->image = std::unique_ptr<im::HybridArray>(nullptr);
            self->dtype = NULL;
        }
        return reinterpret_cast<PyObject*>(self); /// all is well, return self
    }

    /// __init__ implementation
    int NumpyImage_init(NumpyImage* self, PyObject* args, PyObject* kwargs) {
        PyArray_Descr* dtype = NULL;
        char const* filename = NULL;
        char const* keywords[] = { "file", "dtype", NULL };
        static const im::options_map opts; /// not currently used when reading
        
        if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "s|O&", const_cast<char**>(keywords),
            &filename,
            PyArray_DescrConverter, &dtype)) {
                PyErr_SetString(PyExc_ValueError,
                    "bad arguments to NumpyImage_init");
                return -1;
        }
        
        im::ArrayFactory factory;
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
        
        self->image = im::detail::dynamic_cast_unique<im::HybridArray>(std::move(output));
        
        /// ALL IS WELL:
        return 0;
    }
    
    /// __repr__ implementation
    PyObject* NumpyImage_Repr(NumpyImage* im) {
        return PyString_FromFormat("<NumpyImage @ %p>", im);
    }
    
    int NumpyImage_GetBuffer(PyObject* self, Py_buffer* view, int flags) {
        NumpyImage* pyim = reinterpret_cast<NumpyImage*>(self);
        int out = pyim->image->populate_buffer(view,
                                               (NPY_TYPES)pyim->dtype->type_num,
                                               flags);
        // Py_INCREF(self);
        // view->obj = self;
        return out;
    }
    
    void NumpyImage_ReleaseBuffer(PyObject* self, Py_buffer* view) {
        NumpyImage* pyim = reinterpret_cast<NumpyImage*>(self);
        pyim->image->release_buffer(view);
        // Py_XDECREF(view->obj);
        // view->obj = NULL;
        PyBuffer_Release(view);
    }
    
    /// DEALLOCATE
    void NumpyImage_dealloc(NumpyImage* self) {
        self->cleanup();
        self->ob_type->tp_free((PyObject*)self);
    }
    
    /// NumpyImage.datatype getter
    PyObject*    NumpyImage_GET_dtype(NumpyImage* self, void* closure) {
        return Py_BuildValue("O", self->dtype);
    }
    
    PyObject*    NumpyImage_GET_shape(NumpyImage* self, void* closure) {
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
    
} /* namespace (anon.) */


static PyBufferProcs NumpyImage_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)NumpyImage_GetBuffer,
    (releasebufferproc)NumpyImage_ReleaseBuffer,
};

static PyGetSetDef NumpyImage_getset[] = {
    {
        (char*)"dtype",
            (getter)NumpyImage_GET_dtype,
            NULL,
            (char*)"NumpyImage dtype", NULL },
    {
        (char*)"shape",
            (getter)NumpyImage_GET_shape,
            NULL,
            (char*)"NumpyImage shape tuple", NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

// static PyMethodDef NumpyImage_methods[] = {
//     {
//         "as_array",
//             (PyCFunction)NumpyImage_AsArray,
//             METH_VARARGS | METH_KEYWORDS,
//             "Get an ndarray for a NumpyImage" },
//     SENTINEL
// };

static Py_ssize_t NumpyImage_TypeFlags = Py_TPFLAGS_DEFAULT         | 
                                         Py_TPFLAGS_BASETYPE        | 
                                         Py_TPFLAGS_HAVE_NEWBUFFER;

static PyTypeObject NumpyImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "im.NumpyImage",                                            /* tp_name */
    sizeof(NumpyImage),                                         /* tp_basicsize */
    0,                                                          /* tp_itemsize */
    (destructor)NumpyImage_dealloc,                             /* tp_dealloc */
    0,                                                          /* tp_print */
    0,                                                          /* tp_getattr */
    0,                                                          /* tp_setattr */
    0,                                                          /* tp_compare */
    (reprfunc)NumpyImage_Repr,                                  /* tp_repr */
    0,                                                          /* tp_as_number */
    0,                                                          /* tp_as_sequence */
    0,                                                          /* tp_as_mapping */
    0, /*(hashfunc)NumpyImage_Hash,*/                           /* tp_hash */
    0,                                                          /* tp_call */
    0,                                                          /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                      /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                      /* tp_setattro */
    &NumpyImage_Buffer3000Methods,                              /* tp_as_buffer */
    NumpyImage_TypeFlags,                                       /* tp_flags */
    "Python bindings for NumPy Halide bridge",                  /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    0, /*NumpyImage_methods*/                                   /* tp_methods */
    0,                                                          /* tp_members */
    NumpyImage_getset,                                          /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)NumpyImage_init,                                  /* tp_init */
    0,                                                          /* tp_alloc */
    NumpyImage_new,                                             /* tp_new */
    0,                                                          /* tp_free */
    0,                                                          /* tp_is_gc */
    0,                                                          /* tp_bases */
    0,                                                          /* tp_mro */
    0,                                                          /* tp_cache */
    0,                                                          /* tp_subclasses */
    0,                                                          /* tp_weaklist */
    0,                                                          /* tp_del */
    0,                                                          /* tp_version_tag */
    
};

namespace {
    
    PyObject* structcode_parse(PyObject* self, PyObject* args) {
        char const* code;
        
        if (!PyArg_ParseTuple(args, "s", &code)) {
            PyErr_SetString(PyExc_ValueError,
                "cannot get structcode string");
            return NULL;
        }
        
        return Py_BuildValue("O",
            im::detail::structcode_to_dtype(code));
    }
}

static PyMethodDef NumpyImage_module_functions[] = {
    {
        "structcode_parse",
            (PyCFunction)structcode_parse,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples" },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initim(void) {
    PyObject *module;
    
    PyEval_InitThreads();
    if (PyType_Ready(&NumpyImage_Type) < 0) { return; }
    
    module = Py_InitModule3(
        "im.im", NumpyImage_module_functions,
        "libimread python bindings");
    if (module == NULL) { return; }
    
    /// Bring in NumPy
    import_array();
    
    /// Set up PyCImage object
    Py_INCREF(&NumpyImage_Type);
    PyModule_AddObject(module,
        "NumpyImage",
        (PyObject *)&NumpyImage_Type);
}
