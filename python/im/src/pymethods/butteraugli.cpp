
#include <memory>
#include <string>

#define NO_IMPORT_ARRAY
#include "pymethods/butteraugli.hh"

#include "gil.hpp"
#include "detail.hpp"
#include "pycapsule.hpp"

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/hashing.hh>

namespace py {
    
    namespace functions {
        
        using im::Image;
        using butteraugli::comparison_t;
        
        PyObject* butteraugli(PyObject* _nothing_, PyObject* args, PyObject* kwargs) {
            Image* lhs = nullptr;
            Image* rhs = nullptr;
            PyObject* pylhs = nullptr;
            PyObject* pyrhs = nullptr;
            char const* keywords[] = { "lhs", "rhs", nullptr };
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "OO:butteraugli", const_cast<char**>(keywords),
                &pylhs,     /// "lhs", PyCapsule<im::Image, py::ext::ModelBase>
                &pyrhs))    /// "rhs", PyCapsule<im::Image, py::ext::ModelBase>
            {
                return nullptr;
            }
            
            if (!PyCapsule_IsValid(pylhs, nullptr)) {
                PyErr_SetString(PyExc_ValueError,
                    "butteraugli: invalid LHS capsule");
                return nullptr;
            }
            
            if (!PyCapsule_IsValid(pyrhs, nullptr)) {
                PyErr_SetString(PyExc_ValueError,
                    "butteraugli: invalid RHS capsule");
                return nullptr;
            }
            
            lhs = (Image*)PyCapsule_GetPointer(pylhs, nullptr);
            rhs = (Image*)PyCapsule_GetPointer(pyrhs, nullptr);
            
            comparison_t comparison = butteraugli::compare(lhs, rhs);
            
            if (int(comparison) > 10) { /// VERRRRY HACKY dogg I know
                if (comparison == comparison_t::error_images_incomprable) {
                    PyErr_SetString(PyExc_IOError,
                        "butteraugli: images incomparable");
                } else if (comparison == comparison_t::error_unexpected_channel_count) {
                    PyErr_SetString(PyExc_IOError,
                        "butteraugli: wrong channel count encountered");
                } else if (comparison == comparison_t::error_augli_not_buttered) {
                    PyErr_SetString(PyExc_IOError,
                        "butteraugli: internal library error");
                }
                return nullptr;
            }
            
            return py::convert(int(comparison));
        }
        
    }
}