
#include <string>
#include <tuple>
#include "detail.hh"
#include "structcode.hpp"
#include "gil.hh"

namespace py {
    
    namespace detail {
        
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
