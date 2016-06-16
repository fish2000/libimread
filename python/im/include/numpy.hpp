#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_NUMPY_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_NUMPY_HPP_

#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "hybrid.hh"
#include "detail.hpp"
#include "gil.hpp"
#include "typecode.hpp"

namespace py {
    
    namespace detail {
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wswitch"
        template <typename ImageType> inline
        PyObject* image_typed_idx(ImageType const& image,
                                  int tc = NPY_UINT8, std::size_t nidx = 0) {
            switch (tc) {
                case NPY_FLOAT: {
                    float op = static_cast<float*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_DOUBLE:
                case NPY_LONGDOUBLE: {
                    double op = static_cast<double*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_SHORT:
                case NPY_BYTE: {
                    byte op = static_cast<byte*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_USHORT:
                case NPY_UBYTE: {
                    uint8_t op = static_cast<uint8_t*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_INT: {
                    int32_t op = static_cast<int32_t*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_UINT: {
                    uint32_t op = static_cast<uint32_t*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_LONG:
                case NPY_LONGLONG: {
                    int64_t op = static_cast<int64_t*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
                case NPY_ULONG:
                case NPY_ULONGLONG: {
                    uint64_t op = static_cast<uint64_t*>(image->rowp(0))[nidx];
                    return py::convert(op);
                }
                break;
            }
            uint8_t op = image->template rowp_as<uint8_t>(0)[nidx];
            return py::convert(op);
        }
        #pragma clang diagnostic pop
        
    }
    
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
            int nbytes              = image.nbytes();
            int flags               = include_descriptor ? NPY_ARR_HAS_DESCR : 0;
            NPY_TYPES typecode      = image.dtype();
            char typechar           = (char)typecode::typechar(typecode);
            char const* structcode  = im::detail::structcode(typecode);
            PyObject* descriptor    = nullptr;
            
            if (include_descriptor) {
                py::gil::ensure yesgil;
                descriptor = py::detail::structcode_to_dtype(structcode);
            }
            
            out = new PyArrayInterface { 2,
                ndims,  typechar,
                nbytes, flags,
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