
#include "hybridimage.hh"

PyTypeObject HybridImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                                  /* ob_size */
    "im.HybridImage",                                                   /* tp_name */
    sizeof(HybridImage),                                                /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::image::dealloc<HalideNumpyImage>,                   /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    0,                                                                  /* tp_compare */
    (reprfunc)py::image::repr<HalideNumpyImage>,                        /* tp_repr */
    0,                                                                  /* tp_as_number */
    &HybridImage_SequenceMethods,                                       /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::image::hash<HalideNumpyImage>,                        /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::image::str<HalideNumpyImage>,                         /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    &HybridImage_Buffer3000Methods,                                     /* tp_as_buffer */
    HybridImage_TypeFlags,                                              /* tp_flags */
    "Python bindings for NumPy Halide bridge",                          /* tp_doc */
    0,                                                                  /* tp_traverse */
    0,                                                                  /* tp_clear */
    0,                                                                  /* tp_richcompare */
    0,                                                                  /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    HybridImage_methods,                                                /* tp_methods */
    0,                                                                  /* tp_members */
    HybridImage_getset,                                                 /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::image::init<HalideNumpyImage, HybridFactory>,         /* tp_init */
    0,                                                                  /* tp_alloc */
    py::image::createnew<HalideNumpyImage>,                             /* tp_new */
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

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initim(void) {
    /// Allocate a pointer to the module object
    PyObject* module;
    
    /// Initialize Python threads and GIL state
    PyEval_InitThreads();
    
    /// Check readiness of HybridImage type (?)
    if (PyType_Ready(&HybridImage_Type) < 0) { return; }
    
    /// Actually initialize the module object,
    /// setting up the module's external C-function table
    module = Py_InitModule3(
        "im.im", HybridImage_module_functions,
        "libimread python bindings");
    if (module == NULL) { return; }
    
    /// Bring in NumPy's C-API
    import_array();
    
    /// Add the HybridImage type object to the module
    Py_INCREF(&HybridImage_Type);
    PyModule_AddObject(module,
        "HybridImage",
        (PyObject*)&HybridImage_Type);
}
