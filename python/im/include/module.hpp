
#include <Python.h>

#include "hybridimage.hh"
#include "halideimage.hh"
#include "detail.hpp"
#include "hybrid.hh"

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
