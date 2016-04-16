
#include <vector>
#include <algorithm>

#include "detail.hpp"
#include "gil.hpp"
#include "options.hpp"
#include "pycapsule.hpp"
#include "structcode.hpp"

#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

#include <libimread/libimread.hpp>
#include <libimread/imageformat.hh>

namespace py {
    
    namespace capsule {
        
        template <>
        destructor_t decapsulator<void, void> = [](PyObject* capsule) {
            if (!PyCapsule_IsValid(capsule, PyCapsule_GetName(capsule))) {
                PyErr_SetString(PyExc_ValueError,
                    "Invalid PyCapsule");
            }
            char const* name = PyCapsule_GetName(capsule);
            if (name) { std::free((void*)name); name = nullptr;    }
        };
        
    }
    
    PyObject* None()  { return Py_BuildValue("O", Py_None); }
    PyObject* True()  { return Py_BuildValue("O", Py_True); }
    PyObject* False() { return Py_BuildValue("O", Py_False); }
    
    PyObject* boolean(bool truth) {
         return Py_BuildValue("O", truth ? Py_True : Py_False);
    }
    
    #if PY_MAJOR_VERSION < 3
    PyObject* string(std::string const& s) {
        return PyString_FromStringAndSize(s.c_str(), s.size());
    }
    PyObject* string(std::wstring const& w) {
        return PyUnicode_FromWideChar(w.data(), w.size());
    }
    PyObject* string(bytevec_t const& bv) {
        return PyString_FromStringAndSize((char const*)&bv[0], bv.size());
    }
    PyObject* string(char const* s) {
        return PyString_FromString(s);
    }
    PyObject* string(char const* s, std::size_t length) {
        return PyString_FromStringAndSize(s, length);
    }
    PyObject* string(char s) {
        return PyString_FromFormat("%c", s);
    }
    #elif PY_MAJOR_VERSION >= 3
    PyObject* string(std::string const& s) {
        return PyBytes_FromStringAndSize(s.c_str(), s.size());
    }
    PyObject* string(std::wstring const& w) {
        return PyUnicode_FromWideChar(w.data(), w.size());
    }
    PyObject* string(bytevec_t const& bv) {
        return PyBytes_FromStringAndSize((char const*)&bv[0], bv.size());
    }
    PyObject* string(char const* s) {
        return PyBytes_FromString(s);
    }
    PyObject* string(char const* s, std::size_t length) {
        return PyBytes_FromStringAndSize(s, length);
    }
    PyObject* string(char s) {
        return PyBytes_FromFormat("%c", s);
    }
    #endif
    
    PyObject* object(PyObject* arg) {
        return Py_BuildValue("O", arg ? arg : Py_None);
    }
    PyObject* object(PyArray_Descr* arg) {
        return py::object((PyObject*)arg);
    }
    
    PyObject* convert(PyObject* operand)            { return py::object(operand); }
    PyObject* convert(std::nullptr_t operand)       { return Py_BuildValue("O", Py_None); }
    PyObject* convert(bool operand)                 { return Py_BuildValue("O", operand ? Py_True : Py_False); }
    PyObject* convert(void* operand)                { return py::capsule::encapsulate(operand); }
    PyObject* convert(std::size_t operand)          { return PyInt_FromSize_t(operand); }
    PyObject* convert(Py_ssize_t operand)           { return PyInt_FromSsize_t(operand); }
    PyObject* convert(int8_t operand)               { return PyInt_FromSsize_t(static_cast<Py_ssize_t>(operand)); }
    PyObject* convert(int16_t operand)              { return PyInt_FromSsize_t(static_cast<Py_ssize_t>(operand)); }
    PyObject* convert(int32_t operand)              { return PyInt_FromSsize_t(static_cast<Py_ssize_t>(operand)); }
    PyObject* convert(int64_t operand)              { return PyLong_FromLong(operand); }
    // PyObject* convert(int128_t operand)             { return PyLong_FromLongLong(operand); }
    PyObject* convert(uint8_t operand)              { return PyInt_FromSize_t(static_cast<std::size_t>(operand)); }
    PyObject* convert(uint16_t operand)             { return PyInt_FromSize_t(static_cast<std::size_t>(operand)); }
    PyObject* convert(uint32_t operand)             { return PyInt_FromSize_t(static_cast<std::size_t>(operand)); }
    PyObject* convert(uint64_t operand)             { return PyLong_FromUnsignedLong(operand); }
    // PyObject* convert(uint128_t operand)            { return PyLong_FromUnsignedLongLong(operand); }
    PyObject* convert(float operand)                { return PyFloat_FromDouble(static_cast<double>(operand)); }
    PyObject* convert(double operand)               { return PyFloat_FromDouble(operand); }
    PyObject* convert(long double operand)          { return PyFloat_FromDouble(static_cast<double>(operand)); }
    
    #if PY_MAJOR_VERSION < 3
    PyObject* convert(char* operand)                { return PyString_FromString(operand); }
    PyObject* convert(char const* operand)          { return PyString_FromString(operand); }
    PyObject* convert(std::string const& operand)   { return PyString_FromStringAndSize(operand.c_str(), operand.size()); }
    PyObject* convert(char* operand,
                      std::size_t length)           { return PyString_FromStringAndSize(operand, length); }
    PyObject* convert(char const* operand,
                      std::size_t length)           { return PyString_FromStringAndSize(operand, length); }
    PyObject* convert(std::string const& operand,
                      std::size_t length)           { return PyString_FromStringAndSize(operand.c_str(), length); }
    #elif PY_MAJOR_VERSION >= 3
    PyObject* convert(char* operand)                { return PyBytes_FromString(operand); }
    PyObject* convert(char const* operand)          { return PyBytes_FromString(operand); }
    PyObject* convert(std::string const& operand)   { return PyBytes_FromStringAndSize(operand.c_str(), operand.size()); }
    PyObject* convert(char* operand,
                      std::size_t length)           { return PyBytes_FromStringAndSize(operand, length); }
    PyObject* convert(char const* operand,
                      std::size_t length)           { return PyBytes_FromStringAndSize(operand, length); }
    PyObject* convert(std::string const& operand,
                      std::size_t length)           { return PyBytes_FromStringAndSize(operand.c_str(), length); }
    #endif
    
    PyObject* convert(std::wstring const& operand)  { return PyUnicode_FromWideChar(operand.data(), operand.size()); }
    PyObject* convert(std::wstring const& operand,
                      std::size_t length)           { return PyUnicode_FromWideChar(operand.data(), length); }
    PyObject* convert(Py_buffer* operand)           { return PyMemoryView_FromBuffer(operand); }
    
    PyObject* tuplize()                             { return PyTuple_New(0); }
    PyObject* listify()                             { return PyList_New(0);  }
    
    namespace detail {
        
        using structcode::stringvec_t;
        
        PyObject* structcode_to_dtype(char const* code) {
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
                PyTuple_SET_ITEM(tuple, idx, py::tuple(
                    py::string(pairvec[idx].first),
                    py::string(endianized)));
            }
            
            return tuple;
        }
        
        using im::ImageFormat;
        using im::options_map;
        
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
            
            PyObject* list = PyList_New(max);
            
            for (auto it = formats.begin();
                 it != formats.end() && idx < max;
                 ++it) { std::string const& format = *it;
                         if (format.size() > 0) {
                             PyList_SET_ITEM(list, idx,
                                             py::string(format));
                         } ++idx; }
            
            PyObject* out = PyList_AsTuple(list);
            Py_DECREF(list);
            return out;
        }
        
        PyObject* formats_as_infodict(int idx) {
            stringvec_t formats;
            int max = 0;
            
            {
                py::gil::release nogil;
                formats = py::detail::formats_as_vector();
                max = formats.size();
            }
            
            PyObject* infodict = PyDict_New();
            
            for (auto it = formats.begin();
                 it != formats.end() && idx < max;
                 ++it) { std::string const& format = *it;
                         if (format.size() > 0) {
                             options_map opts;
                             {
                                 py::gil::release nogil;
                                 auto format_ptr = ImageFormat::named(format);
                                 opts = format_ptr->get_options();
                             }
                             PyDict_SetItemString(
                                 infodict,
                                 format.c_str(),
                                 py::options::revert(opts));
                         } ++idx; }
            
            return infodict;
        }
    }
    
}
