
#include <tuple>
#include <algorithm>
#include "detail.hpp"
#include "structcode.hpp"
#include "gil.hpp"

#include <libimread/libimread.hpp>
#include <libimread/imageformat.hh>

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
        
        using im::ImageFormat;
        
        stringvec_t& formats_as_vector() {
            static auto DMV = ImageFormat::registry();
            static stringvec_t out(DMV.size());
            static bool listed = false;
            if (!listed) {
                std::transform(DMV.begin(), DMV.end(),
                               std::back_inserter(out),
                            [](auto const& registrant) {
                    return std::string(registrant.first);
                });
                listed = true;
            }
            return out;
        }
        
        PyObject* formats_as_pytuple(int idx) {
            stringvec_t& formats = py::detail::formats_as_vector();
            const int max = formats.size();
            PyObject* tuple = PyTuple_New(max);
            for (auto it = formats.begin();
                 it != formats.end() && idx < max;
                 ++it) { std::string const& format = *it;
                         if (format.size() > 0) {
                             PyTuple_SET_ITEM(tuple, idx,
                                 PyString_FromString(format.c_str()));
                         }
                         ++idx; }
            return tuple;
        }
    }
    
}
