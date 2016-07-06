
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
        using sizevec_t = std::vector<std::size_t>;
        using pysizevec_t = std::vector<Py_ssize_t>;
        using objectvec_t = std::vector<PyObject*>;
        using objecthasher_t = hash::rehasher<PyObject*>;
        
        struct BatchModel : public ModelBase {
            
            struct BatchIterator : public ModelBase {
                
                using iterator_t = typename objectvec_t::iterator;
                using citerator_t = typename objectvec_t::const_iterator;
                
                static PyTypeObject* type_ptr() { return &BatchIterator_Type; }
                
                void* operator new(std::size_t newsize) {
                    PyTypeObject* type = type_ptr();
                    return reinterpret_cast<void*>(type->tp_alloc(type, 0));
                }
                
                void operator delete(void* voidself) {
                    BatchIterator* self = reinterpret_cast<BatchIterator*>(voidself);
                    type_ptr()->tp_free(py::convert(self));
                }
                
                PyObject_HEAD
                // iterator_t begin;
                // iterator_t end;
                citerator_t cbegin;
                citerator_t cend;
                
                BatchIterator() {}
                
                BatchIterator(BatchIterator&& other) noexcept
                    // :begin(std::move(other.begin))
                    // ,end(std::move(other.end))
                    :cbegin(std::move(other.cbegin))
                    ,cend(std::move(other.cend))
                    {}
                
                explicit BatchIterator(BatchModel const& batch) {
                    // begin = std::begin(batch.internal);
                    // end = std::end(batch.internal);
                    cbegin = std::cbegin(batch.internal);
                    cend = std::cend(batch.internal);
                }
                
                explicit BatchIterator(PyObject* batch)
                    :BatchIterator(*reinterpret_cast<BatchModel*>(batch))
                    {}
                
                PyObject* next() {
                    PyObject* out = cbegin != cend ? py::object(*cbegin++) : nullptr;
                    return out;
                }
                
                Py_ssize_t length() {
                    return static_cast<Py_ssize_t>(std::abs(cbegin - cend));
                }
                
                Py_ssize_t remaining() {
                    return static_cast<Py_ssize_t>(cend - cbegin);
                }
                
                static constexpr Py_ssize_t typeflags() {
                    return Py_TPFLAGS_DEFAULT         |
                           Py_TPFLAGS_BASETYPE;
                }
                
                static char const* typestring() { return "im.Batch.Iterator"; }
                static char const* typedoc() { 
                    return "Iterator class for im.Batch\n";
                }
                
            }; /* BatchIterator */
            
            static PyTypeObject* type_ptr() { return &BatchModel_Type; }
            
            void* operator new(std::size_t newsize) {
                void* out = reinterpret_cast<void*>(
                    PyObject_GC_New(BatchModel, type_ptr()));
                PyObject_GC_Track(
                    reinterpret_cast<PyObject*>(out));
                return out;
                // PyTypeObject* type = type_ptr();
                // return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            }
            
            void operator delete(void* voidself) {
                BatchModel* self = reinterpret_cast<BatchModel*>(voidself);
                PyObject_GC_UnTrack(voidself);
                if (self->weakrefs != nullptr) {
                    PyObject_ClearWeakRefs(py::convert(self));
                }
                self->cleanup();
                // PyObject_GC_Del(voidself);
                type_ptr()->tp_free(py::convert(self));
            }
            
            struct Tag {
                struct FromBatch    {};
                struct Concatenate  {};
                struct Repeat       {};
            };
            
            PyObject_HEAD
            PyObject* weakrefs = nullptr;
            objectvec_t internal;
            PyObject* readoptDict = nullptr;
            PyObject* writeoptDict = nullptr;
            bool clean = false;
            
            BatchModel()
                :weakrefs(nullptr)
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {}
            
            BatchModel(BatchModel const& other)
                :weakrefs(nullptr)
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    std::transform(other.internal.begin(),
                                   other.internal.end(),
                                   std::back_inserter(internal),
                                [](PyObject* pyobj) { Py_INCREF(pyobj);
                                                         return pyobj; });
                    PyDict_Update(readoptDict,  other.readoptDict);
                    PyDict_Update(writeoptDict, other.writeoptDict);
                }
            
            BatchModel(BatchModel&& other) noexcept
                :weakrefs(other.weakrefs)
                ,internal(std::move(other.internal))
                ,readoptDict(other.readoptDict)
                ,writeoptDict(other.writeoptDict)
                {
                    other.clean = true;
                }
            
            explicit BatchModel(BatchModel const& basis,
                                BatchModel const& etc)
                :weakrefs(nullptr)
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    std::transform(basis.internal.begin(),
                                   basis.internal.end(),
                                   std::back_inserter(internal),
                                [](PyObject* pyobj) { Py_INCREF(pyobj);
                                                         return pyobj; });
                    std::transform(etc.internal.begin(),
                                   etc.internal.end(),
                                   std::back_inserter(internal),
                                [](PyObject* pyobj) { Py_INCREF(pyobj);
                                                         return pyobj; });
                }
            
            explicit BatchModel(BatchModel const& basis, int repeat)
                :weakrefs(nullptr)
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    for (int idx = 0; idx < repeat; ++idx) {
                        std::transform(basis.internal.begin(),
                                       basis.internal.end(),
                                       std::back_inserter(internal),
                                    [](PyObject* pyobj) { Py_INCREF(pyobj);
                                                             return pyobj; });
                    }
                }
            
            explicit BatchModel(PyObject* basis, PyObject* etc,
                                typename Tag::Concatenate = typename Tag::Concatenate{})
                :BatchModel(*reinterpret_cast<BatchModel*>(basis),
                            *reinterpret_cast<BatchModel*>(etc))
                {}
            
            explicit BatchModel(PyObject* basis, Py_ssize_t repeat,
                                typename Tag::Repeat = typename Tag::Repeat{})
                :BatchModel(*reinterpret_cast<BatchModel*>(basis),
                             static_cast<int>(repeat))
                {}
            
            
            void swap(BatchModel& other) noexcept {
                using std::swap;
                swap(weakrefs,      other.weakrefs);
                swap(internal,      other.internal);
                swap(readoptDict,   other.readoptDict);
                swap(writeoptDict,  other.writeoptDict);
                swap(clean,         other.clean);
            }
            
            void cleanup(bool force = false) {
                if (!clean) {
                    std::for_each(internal.begin(),
                                  internal.end(),
                               [](PyObject* pyobj) { Py_DECREF(pyobj); });
                    internal.clear();
                    Py_CLEAR(readoptDict);
                    Py_CLEAR(writeoptDict);
                    clean = !force;
                }
            }
            
            int vacay(visitproc visit, void* arg) {
                /// NB. this has to visit the internal vector (maybe?!):
                for (PyObject* item : internal) {
                    Py_VISIT(item);
                }
                Py_VISIT(readoptDict);
                Py_VISIT(writeoptDict);
                return 0;
            }
            
            long __hash__() {
                /// Objects held within a Batch must be Python-hashable;
                /// Also the GIL must be held to use std::hash<PyObject*>
                return std::accumulate(internal.begin(),
                                       internal.end(),
                                       internal.size() + (long)this, /// seed
                                       py::ext::objecthasher_t());
            }
            
            Py_ssize_t __len__() {
                return static_cast<Py_ssize_t>(internal.size());
            }
            
            PyObject* __index__(Py_ssize_t idx) {
                if (idx >= internal.size() || idx < 0) {
                    PyErr_SetString(PyExc_IndexError,
                        "__index__(): out of range");
                    return nullptr;
                }
                return py::convert(internal[static_cast<std::size_t>(idx)]);
            }
            
            PyObject* __index__(Py_ssize_t idx, PyObject* value) {
                if (idx >= internal.size() || idx < 0) {
                    PyErr_SetString(PyExc_IndexError,
                        "__index__(): out of range");
                    return nullptr;
                }
                if (!PyObject_CheckBuffer(value)) {
                    PyErr_SetString(PyExc_ValueError,
                        "__index__(): can't assign unbuffered object");
                    return nullptr;
                }
                std::size_t sidx = static_cast<std::size_t>(idx);
                internal[sidx] = value;
                Py_INCREF(value);
                return py::convert(internal[sidx]);
            }
            
            std::string repr_string() {
                /// start with the BatchModel typestring:
                std::string out(BatchModel::typestring());
                
                /// affix the memory address:
                out += "<@" + std::to_string((long)this) + ">(\n";
                
                /// add the string representation of each object:
                std::for_each(internal.begin(),
                              internal.end(),
                          [&](PyObject* pyobj) {
                    
                    /// extract the Python repr object:
                    PyObject* repr = PyObject_Repr(pyobj);
                    
                    /// stringify + concatenate --
                    /// NB: the Python API is soooo not const-correct:
                    out += "  " + std::string(const_cast<char const*>(
                                              PyString_AS_STRING(repr)));
                    
                    /// conditionally append a comma:
                    out += pyobj == internal.back() ? "\n" : ",\n";
                    
                    /// decref the no-longer-necessary Python repr object:
                    Py_DECREF(repr);
                });
                
                /// festoon the end with an indication of the vector length:
                out += ")[" + std::to_string(internal.size()) + "]";
                
                /// ... and return:
                return out;
            }
            
            PyObject* __repr__() {
                return py::convert(repr_string());
            }
            
            bool append(PyObject* obj) {
                if (!PyObject_CheckBuffer(obj)) {
                    PyErr_SetString(PyExc_ValueError,
                        "append(): can't append unbuffered object");
                    return false;
                }
                internal.push_back(obj);
                Py_INCREF(obj);
                return true;
            }
            
            int count(PyObject* obj) {
                py::gil::release nogil;
                return std::count(internal.begin(),
                                  internal.end(), obj);
            }
            
            bool extend(PyObject* iterable) {
                if (!PySequence_Check(iterable)) {
                    PyErr_SetString(PyExc_ValueError,
                        "extend(): iterable sequence required");
                    return false;
                }
                PyObject* sequence = PySequence_Fast(iterable,
                    "extend(): sequence extraction failed");
                int idx = 0, len = PySequence_Fast_GET_SIZE(sequence);
                for (; idx < len; idx++) {
                    /// PySequence_Fast_GET_ITEM() yields a borrowed reference...
                    PyObject* item = PySequence_Fast_GET_ITEM(sequence, idx);
                    if (!PyObject_CheckBuffer(item)) {
                        PyErr_SetString(PyExc_ValueError,
                            "extend(): unbuffered item found");
                        Py_DECREF(sequence);
                        return false;
                    }
                    internal.push_back(item);
                    Py_INCREF(item); /// ... hence this incref
                }
                Py_DECREF(sequence);
                return !PyErr_Occurred();
            }
            
            Py_ssize_t index(PyObject* obj, Py_ssize_t begin = 0,
                                            Py_ssize_t end = -1) {
                /// -1 means "end"
                if (end == -1) { end = internal.size(); }
                auto result = std::find(internal.begin() + begin,
                                        internal.begin() + end, obj);
                if (result == internal.end()) {
                    PyErr_SetString(PyExc_ValueError,
                        "index(): not found");
                    return -1;
                }
                return static_cast<Py_ssize_t>(internal.begin() - result);
            }
            
            bool insert(Py_ssize_t idx, PyObject* obj) {
                if (!PyObject_CheckBuffer(obj)) {
                    PyErr_SetString(PyExc_ValueError,
                        "insert(): can't insert unbuffered object");
                    return false;
                }
                if (idx > internal.size() || idx < 0) {
                    PyErr_SetString(PyExc_IndexError,
                        "insert(): index out of range");
                    return false;
                }
                internal.insert(internal.begin() + idx, obj);
                Py_INCREF(obj);
                return true;
            }
            
            PyObject* pop(Py_ssize_t idx = -1) {
                if (idx == -1) {
                    PyObject* out = internal.back();
                    internal.pop_back();
                    Py_DECREF(out);
                    return out;
                }
                if (idx > internal.size() || idx < -1) {
                    PyErr_SetString(PyExc_IndexError,
                        "pop(): index out of range");
                    return nullptr;
                }
                PyObject* out = internal.at(idx);
                internal.erase(internal.begin() + idx);
                Py_DECREF(out);
                return out;
            }
            
            bool remove(PyObject* obj) {
                auto result = std::find(internal.begin(),
                                        internal.end(), obj);
                if (result == internal.end()) {
                    PyErr_SetString(PyExc_ValueError,
                        "remove(): not found");
                    return false;
                }
                Py_DECREF(internal.at(internal.begin() - result));
                internal.erase(result);
                return true;
            }
            
            void reverse() {
                /// in-place reverse
                py::gil::release nogil;
                std::reverse(internal.begin(), internal.end());
            }
            
            PyObject* as_pylist() {
                PyObject* out = PyList_New(0);
                if (!internal.empty()) {
                    std::for_each(internal.begin(),
                                  internal.end(),
                           [&out](PyObject* pyobj) { PyList_Append(out, pyobj); });
                }
                return out;
            }
            
            PyObject* as_pytuple() {
                PyObject* list = as_pylist();
                PyObject* out = PyList_AsTuple(list);
                Py_DECREF(list);
                return out;
            }
            
            #define NUMBER_FROM_PYOBJECT(obj, attribute)                                            \
                if (PyObject_HasAttrString(obj, #attribute) == 1) {                                 \
                    PyObject* pynumber = PyObject_GetAttrString(obj, #attribute);                   \
                    if (pynumber != nullptr) {                                                      \
                        Py_ssize_t out = PyInt_AsSsize_t(pynumber);                                 \
                        Py_DECREF(pynumber);                                                        \
                        return out;                                                                 \
                    }                                                                               \
                }
            
            static Py_ssize_t width_of(PyObject* obj) {
                NUMBER_FROM_PYOBJECT(obj, width);
                NUMBER_FROM_PYOBJECT(obj, w);
                NUMBER_FROM_PYOBJECT(obj, Width);
                NUMBER_FROM_PYOBJECT(obj, W);
                return 0;
            }
            
            static Py_ssize_t height_of(PyObject* obj) {
                NUMBER_FROM_PYOBJECT(obj, height);
                NUMBER_FROM_PYOBJECT(obj, h);
                NUMBER_FROM_PYOBJECT(obj, Height);
                NUMBER_FROM_PYOBJECT(obj, H);
                return 0;
            }
            
            #undef NUMBER_FROM_PYOBJECT
            
            Py_ssize_t width() {
                if (internal.empty()) {
                    return 0;
                } else if (internal.size() == 1) {
                    return BatchModel::width_of(internal.front());
                }
                pysizevec_t widths;
                Py_ssize_t width_front = BatchModel::width_of(internal.front());
                std::for_each(internal.begin(),
                              internal.end(),
                    [&widths](PyObject* pyobj) { widths.push_back(BatchModel::width_of(pyobj)); });
                bool wat = std::all_of(widths.cbegin(),
                                       widths.cend(),
                         [width_front](Py_ssize_t widthval) { return width_front == widthval; });
                if (!wat) {
                    PyErr_SetString(PyExc_ValueError,
                        "width(): mismatch");
                    return -1;
                }
                return width_front;
            }
            
            Py_ssize_t height() {
                if (internal.empty()) {
                    return 0;
                } else if (internal.size() == 1) {
                    return BatchModel::height_of(internal.front());
                }
                pysizevec_t heights;
                Py_ssize_t height_front = BatchModel::height_of(internal.front());
                std::for_each(internal.begin(),
                              internal.end(),
                   [&heights](PyObject* pyobj) { heights.push_back(BatchModel::height_of(pyobj)); });
                bool wat = std::all_of(heights.cbegin(),
                                       heights.cend(),
                        [height_front](Py_ssize_t heightval) { return height_front == heightval; });
                if (!wat) {
                    PyErr_SetString(PyExc_ValueError,
                        "height(): mismatch");
                    return -1;
                }
                return height_front;
            }
            
            options_map readopts() {
                return py::options::parse(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            bool load(char const* source, options_map const& opts) { return true; }
            bool load(Py_buffer const& view, options_map const& opts) { return true; }
            bool loadfilelike(PyObject* file, options_map const& opts) { return true; }
            bool loadblob(Py_buffer const& view, options_map const& opts) { return true; }
            
            bool save(char const* destination, options_map const& opts) { return true; }
            bool savefilelike(PyObject* file, options_map const& opts) { return true; }
            PyObject* saveblob(options_map const& opts) { return py::None(); }
            
            static constexpr Py_ssize_t typeflags() {
                return Py_TPFLAGS_DEFAULT         |
                       Py_TPFLAGS_BASETYPE        |
                       Py_TPFLAGS_HAVE_GC         |
                       Py_TPFLAGS_HAVE_WEAKREFS;
            }
            
            static char const* typestring() { return "im.Batch"; }
            static char const* typedoc() { 
                return "Python sequence of Python-buffer-enabled objects\n"
                       "from which image batches may be read and/or written\n";
            }
            
        }; /* BatchModel */
        
    } /* namespace ext */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_