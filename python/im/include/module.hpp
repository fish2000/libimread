
#include <Python.h>
#include <numpy/arrayobject.h>

#include "models/models.hh"
#include "pymethods/batchmethods.hh"
#include "pymethods/pymethods.hh"
#include "detail.hpp"
#include "hybrid.hh"

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>

#ifndef PyMODINIT_FUNC
#if PY_VERSION_HEX >= 0x03000000
#define PyMODINIT_FUNC PyObject*
#else
#define PyMODINIT_FUNC void
#endif
#endif
