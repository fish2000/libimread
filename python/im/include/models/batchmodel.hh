
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_

#include <cstring>
#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include <Python.h>
#include <structmember.h>

#include "../buffer.hpp"
#include "../check.hh"
#include "../gil.hpp"
#include "../detail.hpp"
#include "../options.hpp"
#include "../numpy.hpp"
#include "base.hh"

#include <libimread/rehash.hh>


namespace std {
    
    template <>
    struct hash<PyObject*> {
        
        typedef PyObject* argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type pyobj) const {
            /// Who's holding the GIL? We abdicate responsibility;
            /// but so I do declare it should be held by someone
            // py::gil::ensure yesgil;
            return static_cast<result_type>(PyObject_Hash(pyobj));
        }
        
    };
    
} /* namespace std */

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        using objectvec_t = std::vector<PyObject*>;
        using objecthasher_t = hash::rehasher<PyObject*>;
        
        struct BatchModel : public ModelBase {
            
            static PyTypeObject* type_ptr() { return &BatchModel_Type; }
            
            void* operator new(std::size_t newsize) {
                PyTypeObject* type = type_ptr();
                return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            }
            
            void operator delete(void* voidself) {
                BufferModelBase* self = reinterpret_cast<BufferModelBase*>(voidself);
                PyObject* pyself = py::convert(self);
                if (self->weakrefs != nullptr) {
                    PyObject_ClearWeakRefs(pyself);
                }
                self->cleanup();
                type_ptr()->tp_free(pyself);
            }
            
            PyObject_HEAD
            PyObject* weakrefs = nullptr;
            objectvec_t internal;
            PyObject* readoptDict = nullptr;
            PyObject* writeoptDict = nullptr;
            bool clean = false;
            
            BatchModel()
                :internal{ nullptr }
                ,readoptDict(nullptr)
                ,writeoptDict(nullptr)
                {}
            
            BatchModel(BatchModel const& other)
                :internal{ nullptr }
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    std::transform(other.internal.begin(),
                                   other.internal.end(),
                                   std::back_inserter(internal),
                                [](PyObject* pyobj) { Py_INCREF(pyobj);
                                                         return pyobj; });
                }
            
            BatchModel(BatchModel&& other) noexcept
                :internal(std::move(other.internal))
                ,readoptDict(other.readoptDict)
                ,writeoptDict(other.writeoptDict)
                {
                    other.clean = true;
                }
            
            void swap(BatchModel& other) noexcept {
                using std::swap;
                swap(internal,      other.internal);
                swap(readoptDict,   other.readoptDict);
                swap(writeoptDict,  other.writeoptDict);
            }
            
            void cleanup(bool force = false) {
                if (clean || !force) {
                    internal.clear();
                    Py_DECREF(readoptDict);
                    Py_DECREF(writeoptDict);
                } else {
                    Py_CLEAR(readoptDict);
                    Py_CLEAR(writeoptDict);
                    std::for_each(internal.begin(),
                                  internal.end(),
                               [](PyObject* pyobj) { Py_DECREF(pyobj); });
                    internal.clear();
                    clean = !force;
                }
            }
            
            int vacay(visitproc visit, void* arg) {
                /// NB. this has to clear the internal vector (maybe?!) ...
                /// and as such the lambda below has to basically, like,
                /// just reference-capture stuff just like in general, OK:
                std::for_each(internal.begin(),
                              internal.end(),
                          [&](PyObject* pyobj) { Py_VISIT(pyobj); });
                Py_VISIT(readoptDict);
                Py_VISIT(writeoptDict);
                return 0;
            }
            
            long __hash__() {
                /// Objects held within a Batch must be Python-hashable
                return std::accumulate(internal.begin(),
                                       internal.end(),
                                       internal.size(), /// seed
                                       py::ext::objecthasher_t());
            }
            
            Py_ssize_t __len__() {
                return static_cast<Py_ssize_t>(internal.size());
            }
            
            PyObject* __index__(Py_ssize_t idx) {
                return py::convert(internal[static_cast<std::size_t>(idx)]);
            }
            
            options_map readopts() {
                return py::options::parse(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            static constexpr Py_ssize_t typeflags() {
                return Py_TPFLAGS_DEFAULT         |
                       Py_TPFLAGS_BASETYPE        |
                       Py_TPFLAGS_HAVE_GC         |
                       Py_TPFLAGS_HAVE_WEAKREFS   |
                       Py_TPFLAGS_HAVE_NEWBUFFER;
            }
            
            static char const* typestring() { return "im.Batch"; }
            static char const* typedoc() { 
                return "Python sequence of Python-buffer-enabled objects\n"
                       "from which image batches may be read and/or written\n";
            }
            
        }; /* BufferModelBase */
        
    } /* namespace ext */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_