
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_

#include <Python.h>

/// forward type declarations
extern PyTypeObject HybridImageModel_Type;
extern PyTypeObject ImageModel_Type;
extern PyTypeObject BufferModel_Type;
extern PyTypeObject ImageBufferModel_Type;
extern PyTypeObject ArrayModel_Type;
extern PyTypeObject ArrayBufferModel_Type;
extern PyTypeObject BatchModel_Type;

#define HybridImage_Check(op)       (Py_TYPE(op) == &HybridImageModel_Type)
#define BufferModel_Check(op)       (Py_TYPE(op) == &BufferModel_Type)
#define ImageModel_Check(op)        (Py_TYPE(op) == &ImageModel_Type)
#define ImageBufferModel_Check(op)  (Py_TYPE(op) == &ImageBufferModel_Type)
#define ArrayModel_Check(op)        (Py_TYPE(op) == &ArrayModel_Type)
#define ArrayBufferModel_Check(op)  (Py_TYPE(op) == &ArrayBufferModel_Type)
#define BatchModel_Check(op)        (Py_TYPE(op) == &BatchModel_Type)

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_CHECK_HH_