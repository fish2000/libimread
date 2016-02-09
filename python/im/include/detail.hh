
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_

#include <memory>
#include <string>
#include <tuple>

#include <Python.h>
#include <structmember.h>
#include "gil.hh"
#include "structcode.hpp"

namespace im {
    
    namespace detail {
        
        /// XXX: remind me why in fuck did I write this shit originally
        template <typename T, typename pT>
        std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT>&& source) {
            /// Force a dynamic_cast upon a unique_ptr via interim swap
            /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
            /// ... from http://stackoverflow.com/a/14777419/298171
            if (!source) { return std::unique_ptr<T>(); }
            
            /// Throws a std::bad_cast() if this doesn't work out
            T *destination = &dynamic_cast<T&>(*source.get());
            
            source.release();
            std::unique_ptr<T> out(destination);
            return out;
        }
        
        PyObject* structcode_to_dtype(char const* code) {
            using structcode::stringvec_t;
            using structcode::structcode_t;
            using structcode::parse_result_t;
            
            std::string endianness;
            stringvec_t parsetokens;
            structcode_t pairvec;
            Py_ssize_t imax = 0;
            
            {
                py::gil::release nogil;
                std::tie(endianness, parsetokens, pairvec) = structcode::parse(code);
                imax = static_cast<Py_ssize_t>(pairvec.size());
            }
            
            if (!bool(imax)) {
                PyErr_Format(PyExc_ValueError,
                    "Structcode %.200s parsed to zero-length", code);
                return NULL;
            }
            
            /// Make python list of tuples
            PyObject* tuple = PyTuple_New(imax);
            for (Py_ssize_t idx = 0; idx < imax; idx++) {
                PyTuple_SET_ITEM(tuple, idx, PyTuple_Pack(2,
                    PyString_FromString(pairvec[idx].first.c_str()),
                    PyString_FromString((endianness + pairvec[idx].second).c_str())));
            }
            
            return tuple;
        }
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_