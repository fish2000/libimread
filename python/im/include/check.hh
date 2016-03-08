
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_

#include <Python.h>

/// forward type declarations
extern PyTypeObject NumpyImage_Type;
#define NumpyImage_Check(op) (Py_TYPE(op) == &NumpyImage_Type)

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_