
#include <memory>
#include <string>

#define NO_IMPORT_ARRAY
#include "pymethods/butteraugli.hh"

#include "gil.hpp"
#include "detail.hpp"
#include "exceptions.hpp"
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
            comparison_t comparison;
            char const* keywords[] = { "lhs", "rhs", nullptr };
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "OO:butteraugli", const_cast<char**>(keywords),
                &pylhs,     /// "lhs", PyCapsule<im::Image, py::ext::ModelBase>
                &pyrhs))    /// "rhs", PyCapsule<im::Image, py::ext::ModelBase>
            {
                return nullptr;
            }
            
            if (!PyCapsule_IsValid(pylhs, nullptr)) {
                return py::ValueError("butteraugli: invalid LHS capsule");
            }
            
            if (!PyCapsule_IsValid(pyrhs, nullptr)) {
                return py::ValueError("butteraugli: invalid RHS capsule");
            }
            
            lhs = (Image*)PyCapsule_GetPointer(pylhs, nullptr);
            rhs = (Image*)PyCapsule_GetPointer(pyrhs, nullptr);
            
            {
                py::gil::release nogil;
                comparison = butteraugli::compare(lhs, rhs);
            }
            
            if (int(comparison) > 10) { /// VERRRRY HACKY dogg I know
                std::string message;
                if (comparison == comparison_t::error_images_incomprable) {
                    message = "butteraugli: images incomparable";
                } else if (comparison == comparison_t::error_unexpected_channel_count) {
                    message = "butteraugli: wrong channel count encountered";
                } else if (comparison == comparison_t::error_augli_not_buttered) {
                    message = "butteraugli: internal library error";
                }
                return py::IOError(message);
            }
            
            return py::convert(int(comparison));
        }
        
    }
}