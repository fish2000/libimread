
#include "numpyimage.hh"

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initim(void) {
    /// Allocate a pointer to the module object
    PyObject* module;
    
    /// Initialize Python threads and GIL state
    PyEval_InitThreads();
    
    /// Check readiness of NumpyImage type (?)
    if (PyType_Ready(&NumpyImage_Type) < 0) { return; }
    
    /// Actually initialize the module object,
    /// setting up the module's external C-function table
    module = Py_InitModule3(
        "im.im", NumpyImage_module_functions,
        "libimread python bindings");
    if (module == NULL) { return; }
    
    /// Bring in NumPy's C-API
    import_array();
    
    /// Add the NumpyImage type object to the module
    Py_INCREF(&NumpyImage_Type);
    PyModule_AddObject(module,
        "NumpyImage",
        (PyObject*)&NumpyImage_Type);
}
