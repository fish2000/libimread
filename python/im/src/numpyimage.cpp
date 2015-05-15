
#include <memory>
#include <string>

#include <Python.h>
#include <structmember.h>

#include "numpy.hh"
#include <libimread/tools.hh>

using ImagePtr = std::unique_ptr<im::HybridArray>;

struct NumpyImage {
    PyObject_HEAD
    ImagePtr image = ImagePtr(nullptr);
    PyArray_Descr *dtype = NULL;
    
    void cleanup() {
        image.release();
        if (dtype) {
            PyObject_Del(dtype);
            dtype = NULL;
        }
    }
    ~NumpyImage() { cleanup(); }
};

/// ALLOCATE / __new__ implementation
static PyObject *NumpyImage_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    NumpyImage *self;
    self = reinterpret_cast<NumpyImage *>(type->tp_alloc(type, 0));
    /// initialize with defaults
    if (self != NULL) {
        self->image = ImagePtr(nullptr);
        self->dtype = NULL;
    }
    return reinterpret_cast<PyObject *>(self); /// all is well, return self
}

/// __init__ implementation
static int NumpyImage_init(NumpyImage *self, PyObject *args, PyObject *kwargs) {
    PyArray_Descr *dtype = NULL;
    const char *filename = NULL;
    static char *keywords[] = { "file", "dtype", NULL };
    static const im::options_map opts; /// not currently used when reading
    
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "s|O&", keywords,
        &filename,
        PyArray_DescrConverter, &dtype)) {
            PyErr_SetString(PyExc_ValueError,
                "bad arguments to NumpyImage_init");
            return -1;
    }
    
    if (dtype) {
        self->dtype = dtype;
    } else {
        self->dtype = PyArray_DescrFromType(NPY_UINT8);
    }
    Py_INCREF(self->dtype);
    
    if (!filename) {
        PyErr_SetString(PyExc_ValueError,
            "No filename");
        return -1;
    }
    
    im::ArrayFactory factory;
    std::unique_ptr<im::ImageFormat> format(im::for_filename(filename));
    std::unique_ptr<im::FileSource> input(new im::FileSource(filename));
    std::unique_ptr<im::Image> output = format->read(input.get(), &factory, opts);
    self->image = im::dynamic_cast_unique<im::HybridArray>(output);
    
    return 0;
}

/// __repr__ implementations
static PyObject *NumpyImage_Repr(NumpyImage *im) {
    return PyString_FromFormat("<NumpyImage @ %p>", im);
}

static const char *NumpyImage_ReprCString(NumpyImage *im) {
    PyObject *out = NumpyImage_Repr(im);
    const char *outstr = PyString_AS_STRING(out);
    Py_DECREF(out);
    return outstr;
}

static std::string NumpyImage_ReprString(NumpyImage *im) {
    return std::string(NumpyImage_ReprCString(im));
}

/// DEALLOCATE
static void NumpyImage_dealloc(NumpyImage *self) {
    self->cleanup();
    self->ob_type->tp_free((PyObject *)self);
}

/// NumpyImage.datatype getter
static PyObject     *NumpyImage_GET_dtype(NumpyImage *self, void *closure) {
    return Py_BuildValue("O", self->dtype);
}

static PyGetSetDef NumpyImage_getset[] = {
    {
        "dtype",
            (getter)NumpyImage_GET_dtype,
            NULL,
            "NumpyImage dtype", NULL },
    SENTINEL
};

// static PyMethodDef NumpyImage_methods[] = {
//     {
//         "as_array",
//             (PyCFunction)NumpyImage_AsArray,
//             METH_VARARGS | METH_KEYWORDS,
//             "Get an ndarray for a NumpyImage" },
//     SENTINEL
// };

static Py_ssize_t NumpyImage_TypeFlags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

static PyTypeObject NumpyImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                          /* ob_size */
    "_im.NumpyImage",                                            /* tp_name */
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
    0,                                                          /* tp_as_buffer */
    NumpyImage_TypeFlags,                                       /* tp_flags */
    "Python bindings for NumPy Halide bridge",                  /* tp_doc */
    0,                                                          /* tp_traverse */
    0,                                                          /* tp_clear */
    0,                                                          /* tp_richcompare */
    0,                                                          /* tp_weaklistoffset */
    0,                                                          /* tp_iter */
    0,                                                          /* tp_iternext */
    0, /*NumpyImage_methods*/,                                  /* tp_methods */
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
};


PyMODINIT_FUNC init_im(void) {
    PyObject *module;
    
    PyEval_InitThreads();
    if (PyType_Ready(&NumpyImage_Type) < 0) { return; }

    module = Py_InitModule3(
        "im._im", NULL,
        "libimread python bindings");
    if (module == None) { return; }
    
    /// Bring in NumPy
    import_array();

    /// Set up PyCImage object
    Py_INCREF(&NumpyImage_Type);
    PyModule_AddObject(module,
        "NumpyImage",
        (PyObject *)&NumpyImage_Type);
}


