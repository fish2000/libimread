
#include <Python.h>

namespace py {
    
    namespace flattery {
        
        PyObject* unflatten(PyObject*, PyObject*);
        PyObject* flatten(PyObject*);
        PyObject* flatten_mappings(PyObject*);
        
    }
    
}

