
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_

#include <Python.h>

/// forward type declarations
extern PyTypeObject HybridImage_Type;
#define HybridImage_Check(op) (Py_TYPE(op) == &HybridImage_Type)

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_