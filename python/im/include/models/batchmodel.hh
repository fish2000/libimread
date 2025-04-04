
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BATCHMODEL_HH_

#include <cstring>
#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <unordered_map>
// #include <Python.h>
// #include <structmember.h>

#include "base.hh"
#include "../buffer.hpp"
#include "../check.hh"
#include "../gil.hpp"
// #include "../detail.hpp"
#include "../exceptions.hpp"
#include "../options.hpp"
#include "../bufferview.hpp"
// #include "base.hh"

#include <libimread/rehash.hh>
#include <libimread/image.hh>
#include <libimread/imagelist.hh>

namespace py {
    
    namespace ext {
        
        namespace detail {
            
            template <typename T>
            std::string tohex(T i) {
                std::stringstream stream;
                stream << "0x" 
                       << std::setfill('0') << std::setw(sizeof(T) * 2)
                       << std::hex << i;
                return stream.str();
            }
            
        }
        
        using im::byte;
        using im::Options;
        using im::Image;
        using im::ImageList;
        using ImageBufferView = im::buffer::View;
        
        using sizevec_t = std::vector<std::size_t>;
        using pysizevec_t = std::vector<Py_ssize_t>;
        using objectvec_t = std::vector<PyObject*>;
        using objecthasher_t = hash::rehasher<PyObject*>;
        
        struct BatchModel : public ModelBase<BatchModel, true> {
            
            struct BatchIterator : public ModelBase<BatchModel> {
                
                using iterator_t = typename objectvec_t::iterator;
                using citerator_t = typename objectvec_t::const_iterator;
                
                static PyTypeObject* type_ptr() { return &BatchIterator_Type; }
                
                // void* operator new(std::size_t newsize) {
                //     PyTypeObject* type = type_ptr();
                //     return reinterpret_cast<void*>(type->tp_alloc(type, 0));
                // }
                //
                // void operator delete(void* voidself) {
                //     BatchIterator* self = reinterpret_cast<BatchIterator*>(voidself);
                //     type_ptr()->tp_free(py::convert(self));
                // }
                
                PyObject_HEAD
                citerator_t cbegin;
                citerator_t cend;
                
                BatchIterator() {}
                
                BatchIterator(BatchIterator&& other) noexcept
                    :cbegin(std::move(other.cbegin))
                    ,cend(std::move(other.cend))
                    {}
                
                explicit BatchIterator(BatchModel const& batch) {
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
                           Py_TPFLAGS_BASETYPE        |
                           Py_TPFLAGS_HAVE_ITER;
                }
                
                static char const* typestring() { return "im.Batch.Iterator"; }
                static char const* typedoc() { 
                    return "Iterator class for im.Batch\n";
                }
                
            }; /* BatchIterator */
            
            static PyTypeObject* type_ptr() { return &BatchModel_Type; }
            
            // void* operator new(std::size_t newsize) {
            //     // PyTypeObject* type = type_ptr();
            //     // return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            //     void* out = reinterpret_cast<void*>(
            //         PyObject_GC_New(BatchModel, type_ptr()));
            //     PyObject_GC_Track(
            //         reinterpret_cast<PyObject*>(out));
            //     return out;
            // }
            //
            // void operator delete(void* voidself) {
            //     BatchModel* self = reinterpret_cast<BatchModel*>(voidself);
            //     PyObject_GC_UnTrack(voidself);
            //     if (self->weakrefs != nullptr) {
            //         PyObject_ClearWeakRefs(py::convert(self));
            //     }
            //     self->cleanup();
            //     // PyObject_GC_Del(voidself);
            //     type_ptr()->tp_free(py::convert(self));
            // }
            
            struct Tag {
                struct FromBatch            {};
                struct Concatenate          {};
                struct Repeat               {};
                struct Slice                {};
                struct KeyCallable          {};
                struct ComparisonCallable   {};
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
            
            explicit BatchModel(BatchModel const& basis, int start, int end)
                :weakrefs(nullptr)
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    using std::swap;
                    std::size_t size = basis.internal.size();
                    if (start < 0)   { start = size - std::abs(start); }
                    if (end < 0)     { end   = size - std::abs(end); }
                    if (end < start) { swap(end, start); }
                    if (start < end && start < size && end < size) {
                        std::transform(basis.internal.begin() + start,
                                       basis.internal.begin() + end,
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
            
            explicit BatchModel(PyObject* basis, Py_ssize_t start, Py_ssize_t end,
                                typename Tag::Slice = typename Tag::Slice{})
                :BatchModel(*reinterpret_cast<BatchModel*>(basis),
                             static_cast<int>(start),
                             static_cast<int>(end))
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
                if (!clean || force) {
                    if (!internal.empty()) {
                        std::for_each(internal.begin(),
                                      internal.end(),
                                   [](PyObject* pyobj) { Py_DECREF(pyobj); });
                        internal.clear();
                    }
                    Py_CLEAR(readoptDict);
                    Py_CLEAR(writeoptDict);
                    clean = !force;
                }
            }
            
            int vacay(visitproc visit, void* arg) {
                /// NB. this has to visit the internal vector (maybe?!):
                if (!internal.empty()) {
                    for (PyObject* item : internal) {
                        if (item) {
                            Py_VISIT(item);
                        }
                    }
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
                    return py::IndexError("__index__(): out of range");
                }
                return py::convert(internal[static_cast<std::size_t>(idx)]);
            }
            
            PyObject* __index__(Py_ssize_t idx, PyObject* value) {
                if (idx >= internal.size() || idx < 0) {
                    return py::IndexError("__index__(): out of range");
                }
                if (!PyObject_CheckBuffer(value)) {
                    return py::ValueError("__index__(): can't assign unbuffered object");
                }
                std::size_t sidx = static_cast<std::size_t>(idx);
                py::ref old = internal.at(sidx);
                internal[sidx] = value;
                Py_INCREF(internal[sidx]);
                return value;
            }
            
            std::string repr_string() {
                /// start with the BatchModel typestring:
                std::string out(BatchModel::typestring());
                out += internal.empty() ? "(" : "(\n";
                
                /// add the string representation of each object:
                std::for_each(internal.begin(),
                              internal.end(),
                          [&](PyObject* pyobj) {
                    
                    /// extract the Python repr object:
                    py::ref repr = PyObject_Repr(pyobj);
                    
                    /// stringify + concatenate --
                    out += "    " + repr.to_string();
                    
                    /// conditionally append a comma:
                    out += pyobj == internal.back() ? "\n" : ",\n";
                });
                
                /// festoon the end with an indication of the vector length:
                out += ")[" + std::to_string(internal.size()) + "] @ <";
                
                /// affix the hexadecimal memory address:
                out += detail::tohex((std::ptrdiff_t)this) + ">";
                
                /// ... and return:
                return out;
            }
            
            PyObject* __repr__() {
                return py::convert(repr_string());
            }
            
            PyObject* __iter__() {
                return py::convert(new BatchIterator(py::convert(this)));
            }
            
            bool append(PyObject* obj) {
                if (!PyObject_CheckBuffer(obj)) {
                    return py::ValueError("append(): can't append unbuffered object", false);
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
                    return py::ValueError("extend(): iterable sequence required", false);
                }
                py::ref sequence = PySequence_Fast(iterable, "extend(): sequence extraction failed");
                int idx = 0, len = PySequence_Fast_GET_SIZE(sequence.get());
                for (; idx < len; idx++) {
                    /// PySequence_Fast_GET_ITEM() yields a borrowed reference...
                    PyObject* item = PySequence_Fast_GET_ITEM(sequence.get(), idx);
                    if (!PyObject_CheckBuffer(item)) {
                        return py::ValueError("extend(): unbuffered item found", false);
                    }
                    internal.push_back(item);
                    Py_INCREF(item); /// ... hence this incref
                }
                return !py::ErrorOccurred();
            }
            
            Py_ssize_t index(PyObject* obj, Py_ssize_t begin = 0,
                                            Py_ssize_t end = -1) {
                /// -1 means "end"
                using std::swap;
                std::size_t size = internal.size();
                if (begin < 0)   { begin = size - std::abs(begin); }
                if (end < 0)     { end   = size - std::abs(end); }
                if (end < begin) { swap(end, begin); }
                if (begin < end && begin < size && end < size) {
                    auto result = std::find(internal.begin() + begin,
                                            internal.begin() + end, obj);
                    if (result == internal.end()) {
                        return py::ValueError("index(): not found", -1);
                    }
                    return std::abs(static_cast<Py_ssize_t>(internal.begin() - result));
                } else {
                    return py::ValueError("index(): bad indices", -1);
                }
            }
            
            bool contains(PyObject* obj) {
                py::gil::release nogil;
                if (obj == nullptr) { return false; }
                auto result = std::find(internal.begin(),
                                        internal.end(), obj);
                return result != internal.end();
            }
            
            bool insert(Py_ssize_t idx, PyObject* obj) {
                if (!PyObject_CheckBuffer(obj)) {
                    return py::ValueError("insert(): can't insert unbuffered object", false);
                }
                if (idx > internal.size() || idx < 0) {
                    return py::IndexError("insert(): index out of range", false);
                }
                internal.insert(internal.begin() + idx, obj);
                Py_INCREF(obj);
                return true;
            }
            
            PyObject* pop(Py_ssize_t idx = -1) {
                /// NOTE: NO DECREFS! OR INCREFS!!
                /// ... objects are passed on straight from
                /// the internal vector out to the interpreter!
                /// ... IT WORKS!!!!!!!!!!!!!!!
                if (idx == -1) {
                    PyObject* out = internal.back();
                    internal.pop_back();
                    return out;
                }
                if (idx > internal.size() || idx < -1) {
                    return py::IndexError("pop(): index out of range");
                }
                PyObject* out = internal.at(idx);
                internal.erase(internal.begin() + idx);
                return out;
            }
            
            bool remove(PyObject* obj) {
                auto result = std::find(internal.begin(),
                                        internal.end(), obj);
                if (result == internal.end()) {
                    return py::ValueError("remove(): not found", false);
                }
                Py_DECREF(*result);
                internal.erase(result);
                return true;
            }
            
            void reverse() {
                /// in-place reverse
                py::gil::release nogil;
                std::reverse(internal.begin(), internal.end());
            }
            
            using pyptr_t = std::add_pointer_t<PyObject>;
            using comparison_f = std::function<bool(pyptr_t, pyptr_t)>;
            using keymap_t = std::unordered_map<pyptr_t, pyptr_t>;
            
            bool sort() {
                /// in-place stable sort
                std::stable_sort(internal.begin(),
                                 internal.end());
                return true;
            }
            
            bool sort(comparison_f&& comparison) {
                /// in-place stable sort
                py::gil::ensure yesgil;
                std::stable_sort(internal.begin(),
                                 internal.end(),
                                 std::forward<comparison_f>(comparison));
                return true;
            }
            
            bool sort(PyObject* cmp, typename Tag::ComparisonCallable) {
                /// 'cmp' must be a callable like:
                /// def compare(x, y):
                ///     if x == y:
                ///         return 0
                ///     elif x < y:
                ///         return -1
                ///     else: # x > y
                ///         return 1
                if (PyCallable_Check(cmp)) {
                    return sort([&cmp](pyptr_t x, pyptr_t y) -> bool {
                        py::ref result = PyObject_CallFunctionObjArgs(cmp, x, y, nullptr);
                        return PyInt_AsLong(result) < 0L;
                    });
                } else {
                    return false;
                }
            }
            
            bool sort(PyObject* key, typename Tag::KeyCallable = typename Tag::KeyCallable{}) {
                /// 'key' must be a callable like:
                /// lambda thing: numpy.average(thing.entropy()) # or what have you
                /// ... as in, called with every PyObject* in internal
                if (PyCallable_Check(key)) {
                    /// make unique-ifed copy of internal vector:
                    objectvec_t uniques(internal);
                    std::sort(uniques.begin(), uniques.end());
                    auto last = std::unique(uniques.begin(), uniques.end());
                    uniques.erase(last, uniques.end());
                    
                    /// fill keymap, calling key function exactly once per object:
                    keymap_t keymap;
                    std::transform(uniques.begin(), uniques.end(),
                                   std::inserter(keymap, keymap.begin()),
                            [&key](PyObject* obj) -> keymap_t::value_type {
                        PyObject* value = PyObject_CallFunctionObjArgs(key, obj, nullptr);
                        return std::make_pair(obj, value);
                    });
                    
                    /// do the sort:
                    bool out = sort([&keymap](pyptr_t x, pyptr_t y) -> bool {
                        return PyObject_RichCompareBool(keymap[x], keymap[y], Py_LT) == 1;
                    });
                    
                    /// clean up the mapped values:
                    for (auto const& objpair : keymap) {
                        Py_XDECREF(objpair.second);
                    }
                    
                    /// return boolean:
                    return out;
                } else {
                    return false;
                }
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
                py::ref list = as_pylist();
                return PyList_AsTuple(list);
            }
            
            #define NUMBER_FROM_PYOBJECT(obj, attribute)                                            \
                if (PyObject_HasAttrString(obj, #attribute) == 1) {                                 \
                    py::ref pynumber = PyObject_GetAttrString(obj, #attribute);                     \
                    if (!pynumber.empty()) {                                                        \
                        return PyInt_AsSsize_t(pynumber);                                           \
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
                    return py::ValueError("width(): mismatch", -1);
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
                    return py::ValueError("height(): mismatch", -1);
                }
                return height_front;
            }
            
            ImageList as_imagelist() {
                static const int flags = PyBUF_ND | PyBUF_STRIDES;
                ImageList out;
                
                /// double-check bufferage
                bool buffered = std::all_of(internal.begin(),
                                            internal.end(),
                                         [](PyObject* obj) {
                    return bool(PyObject_CheckBuffer(obj));
                });
                
                if (!buffered) {
                    /// FREAK OUT!!!
                    return out;
                }
                
                Py_buffer view{ 0 };
                std::for_each(internal.begin(),
                              internal.end(),
                          [&](PyObject* obj) {
                    if (PyObject_GetBuffer(obj, &view, flags) != -1) {
                        out.push_back(new ImageBufferView(&view));
                        PyBuffer_Release(&view);
                    }
                });
                
                return out;
            }
            
            Options readopts() {
                return py::options::parse(readoptDict);
            }
            
            Options writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            bool load(char const* source, Options const& opts) { return true; }
            bool load(Py_buffer const& view, Options const& opts) { return true; }
            bool loadfilelike(PyObject* file, Options const& opts) { return true; }
            bool loadblob(Py_buffer const& view, Options const& opts) { return true; }
            
            bool save(char const* destination, Options const& opts) {
                std::unique_ptr<ImageFormat> format;
                Options default_opts;
                bool exists = false,
                     can_write = false,
                     overwrite = true;
                
                try {
                    py::gil::release nogil;
                    format = im::for_filename(destination);
                    can_write = format->format_can_write_multi();
                    exists = path::exists(destination);
                    overwrite = opts.cast<bool>("overwrite", overwrite);
                    if (can_write && exists && overwrite) {
                        path::remove(destination);
                    }
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError(std::string("Can't find I/O format for file: ") + destination, false);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError("Unimplemented write_multi() in I/O format: " + mime, false);
                }
                
                if (exists && !overwrite) {
                    return py::ValueError(std::string("File exists (opts['overwrite'] == False): ") + destination, false);
                }
                
                ImageList input = as_imagelist();
                
                if (input.size() != internal.size()) {
                    return py::ValueError("Error converting batch contents to ImageList", false);
                }
                
                {
                    py::gil::release nogil;
                    std::unique_ptr<FileSink> output(new FileSink(destination));
                    default_opts = format->add_options(opts);
                    format->write_multi(input, output.get(), default_opts);
                }
                
                return true;
            }
            
            bool savefilelike(PyObject* file, Options const& opts) {
                std::unique_ptr<ImageFormat> format;
                typename py::gil::with::sink_t output;
                std::string ext;
                Options default_opts;
                bool can_write = false;
                
                if (!opts.has("format")) {
                    return py::AttributeError("Output format unspecified", false);
                }
                
                ImageList input = as_imagelist();
                
                if (input.size() != internal.size()) {
                    return py::ValueError("Error converting batch contents to ImageList", false);
                }
                
                try {
                    py::gil::with iohandle(file);
                    output = iohandle.sink();
                    ext = opts.cast<std::string>("format");
                    format = im::get_format(ext.c_str());
                    can_write = format->format_can_write_multi();
                    if (can_write) {
                        default_opts = format->add_options(opts);
                        format->write_multi(input, output.get(), default_opts);
                        return true;
                    }
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError("Can't find I/O format: " + ext, false);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError("Unimplemented write_multi() in I/O format: " + mime, false);
                }
                return false;
            }
            
            PyObject* saveblob(Options const& opts) {
                std::unique_ptr<ImageFormat> format;
                std::vector<byte> data;
                std::string ext, pth;
                Options default_opts;
                bool can_write = false,
                     exists = false,
                     removed = false,
                     as_url = false,
                     as_html = false;
                
                if (!opts.has("format")) {
                    return py::AttributeError("Output format unspecified");
                }
                
                try {
                    py::gil::release nogil;
                    ext = opts.cast<std::string>("format");
                    format = im::get_format(ext.c_str());
                    can_write = format->format_can_write_multi();
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError("Can't find I/O format: " + ext);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError("Unimplemented write_multi() in I/O format: " + mime);
                }
                
                ImageList input = as_imagelist();
                
                if (input.size() != internal.size()) {
                    return py::ValueError("Error converting batch contents to ImageList");
                }
                
                NamedTemporaryFile tf(format->get_suffix(true), false);
                
                {
                    py::gil::release nogil;
                    pth = std::string(tf.filepath.make_absolute().str());
                    tf.filepath.remove();
                    auto output = std::make_unique<FileSink>(pth.c_str());
                    default_opts = format->add_options(opts);
                    format->write_multi(input, output.get(), default_opts);
                    exists = path::exists(pth);
                }
                
                if (!exists) {
                    return py::IOError("Temporary file is AWOL");
                }
                
                {
                    py::gil::release nogil;
                    auto readback = std::make_unique<FileSource>(pth.c_str());
                    data = readback->full_data();
                    tf.close();
                    removed = tf.remove();
                }
                
                if (!removed) {
                    return py::IOError("Failed to remove temporary file");
                }
                
                as_html = opts.cast<bool>("as_html", as_html);
                as_url = opts.cast<bool>("as_url", as_url);
                if (!as_url) { return py::string(data); }
                
                std::string out("data:");
                {
                    py::gil::release nogil;
                    if (as_url) {
                        out += format->get_mimetype() + ";base64,";
                        out += im::base64::encode(&data[0], data.size());
                        if (as_html) {
                            out = std::string("<img src='") + out + "'>";
                        }
                    }
                }
                return py::string(out);
            }
            
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