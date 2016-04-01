
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
            
            if (imax < 1) {
                PyErr_Format(PyExc_ValueError,
                    "Structcode %.200s parsed to zero-length", code);
                return NULL;
            }
            
            /// Make python list of tuples
            PyObject* tuple = PyTuple_New(imax);
            for (Py_ssize_t idx = 0; idx < imax; idx++) {
                std::string endianized(endianness + pairvec[idx].second);
                PyTuple_SET_ITEM(tuple, idx, PyTuple_Pack(2,
                    PyString_FromStringAndSize(pairvec[idx].first.c_str(),
                                               pairvec[idx].first.size()),
                    PyString_FromStringAndSize(endianized.c_str(),
                                               endianized.size())));
            }
            
            return tuple;
        }
        
        using im::ImageFormat;
        
        stringvec_t& formats_as_vector() {
            static auto DMV = ImageFormat::registry();
            static stringvec_t out;
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
            PyObject* list = PyList_New(0);
            stringvec_t formats;
            int max = 0;
            {
                py::gil::release nogil;
                formats = py::detail::formats_as_vector();
                max = formats.size();
            }
            
            /// Creating a zero-size list and then iteratively using
            /// PyList_Append on it (per each of the vector items),
            /// all in order to finish off by handing off this lists' tuple-ization
            /// however PyList_AsTuple() might see it -- well shit, you
            /// might say it's an “unhot loop”. Maybe. Or “functionally subcritical”
            /// perhaps, you say. “Academic”, “past inelegant into the realm of fugly,”
            /// and “amongst the least-fast ways to possibly do it” are also thigs one
            /// perchance might observe about this code. But you kmow what? It only runs
            /// seriously beyond infrequently and probably only once at all ever --
            /// at most once per module load (and generally the information has stopped
            /// changing once the libimread dynamic-loader binary has been initialized
            /// which did you know that's even less frequent?) -- so dogg I am actually
            /// totally cool with it
            
            for (auto it = formats.begin();
                 it != formats.end() && idx < max;
                 ++it) { std::string const& format = *it;
                         if (format.size() > 0) {
                             PyList_Append(list,
                                 PyString_FromStringAndSize(
                                     format.c_str(),
                                     format.size()));
                         } ++idx; }
            return PyList_AsTuple(list);
        }
    }
    
}
