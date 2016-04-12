#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_NUMPY_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_NUMPY_HPP_

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include "hybrid.hh"
#include "detail.hpp"
#include "gil.hpp"
#include "typecode.hpp"

namespace py {
    
    namespace numpy {
        
        // tuple((),()...)  py::detail::structcode_to_dtype(char)
        //   "NPY_UINT32"   typecode::name(NPY_TYPES)
        //            "b"   typecode::typechar(NPY_TYPES)
        //            "b"   im::detail::character_for<PixelType>()
        //            "B"   im::detail::structcode(NPY_TYPES)
        //        HalType   im::detail::for_dtype(NPY_TYPES)
        //        HalType   im::detail::for_type<uint32_t>()
        //          "|b8"   im::detail::encoding_for<PixelType>(e)
        //      NPY_TYPES   im::detail::for_nbits(nbits, is_signed=false)
        // py::detail::structcode_to_dtype(typecode) || nullptr
        
        template <typename ImageType> inline
        PyArrayInterface* array_struct(ImageType const& image,
                                       bool include_descriptor = true) {
            PyArrayInterface* out   = nullptr;
            void* data              = image.rowp(0);
            int ndims               = image.ndims();
            int flags               = include_descriptor ? NPY_ARR_HAS_DESCR : 0;
            NPY_TYPES typecode      = image.dtype();
            NPY_TYPECHAR typechar   = typecode::typechar(typecode);
            PyObject* descriptor    = nullptr;
            
            if (include_descriptor) {
                py::gil::ensure yesgil;
                descriptor = py::detail::structcode_to_dtype(
                             im::detail::structcode(typecode));
            }
            
            out = new PyArrayInterface {
                2,                          /// brought to you by
                ndims,
                (char)typechar,
                sizeof(uint8_t),            /// need to not hardcode this
                flags,
                new Py_intptr_t[ndims],     /// shape
                new Py_intptr_t[ndims],     /// strides
                data,                       /// void* data
                descriptor                  /// PyObject*
            };
            
            for (int idx = 0; idx < ndims; idx++) {
                out->shape[idx]   = image.dim(idx);
                out->strides[idx] = image.stride(idx);
            }
            
            return out;
        }
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_NUMPY_HPP_