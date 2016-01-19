
#include "numpyimage.hh"

using im::HybridArray;
using im::ArrayFactory;
using py::image::NumpyImage;

static PyBufferProcs NumpyImage_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)py::image::getbuffer<NumpyImage>,
    (releasebufferproc)py::image::releasebuffer<NumpyImage>,
};

static PyGetSetDef NumpyImage_getset[] = {
    {
        (char*)"dtype",
            (getter)py::image::get_dtype<NumpyImage>,
            NULL,
            (char*)"NumpyImage dtype", NULL },
    {
        (char*)"shape",
            (getter)py::image::get_shape<NumpyImage>,
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
    0,                                                                  /* ob_size */
    "im.NumpyImage",                                                    /* tp_name */
    sizeof(NumpyImage),                                                 /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::image::dealloc<NumpyImage>,                         /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    0,                                                                  /* tp_compare */
    (reprfunc)py::image::repr<NumpyImage>,                              /* tp_repr */
    0,                                                                  /* tp_as_number */
    0,                                                                  /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::image::hash<NumpyImage>,                              /* tp_hash */
    0,                                                                  /* tp_call */
    0,                                                                  /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    &NumpyImage_Buffer3000Methods,                                      /* tp_as_buffer */
    NumpyImage_TypeFlags,                                               /* tp_flags */
    "Python bindings for NumPy Halide bridge",                          /* tp_doc */
    0,                                                                  /* tp_traverse */
    0,                                                                  /* tp_clear */
    0,                                                                  /* tp_richcompare */
    0,                                                                  /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    0, /*NumpyImage_methods*/                                           /* tp_methods */
    0,                                                                  /* tp_members */
    NumpyImage_getset,                                                  /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::image::init<HybridArray, ArrayFactory, NumpyImage>,   /* tp_init */
    0,                                                                  /* tp_alloc */
    py::image::createnew<HybridArray, NumpyImage>,                      /* tp_new */
    0,                                                                  /* tp_free */
    0,                                                                  /* tp_is_gc */
    0,                                                                  /* tp_bases */
    0,                                                                  /* tp_mro */
    0,                                                                  /* tp_cache */
    0,                                                                  /* tp_subclasses */
    0,                                                                  /* tp_weaklist */
    0,                                                                  /* tp_del */
    0,                                                                  /* tp_version_tag */
    
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

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initim(void) {
    PyObject* module;
    
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
