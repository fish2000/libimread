
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BATCHMETHODS_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BATCHMETHODS_HH_

#include <memory>
#include <string>
#include <Python.h>
#include <structmember.h>

#include "../check.hh"
#include "../detail.hpp"

#include "typecheck.hh"
#include "../models/batchmodel.hh"

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        
        namespace batch {
            
            namespace iterator {
                
                PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                    return py::convert(new BatchIterator());
                }
                
                int init(PyObject* self, PyObject* args, PyObject* kwargs) {
                    // BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    return 0;
                }
                
                /// DEALLOCATE
                void dealloc(PyObject* self) {
                    BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    delete iterator;
                }
                
                PyObject* next(PyObject* self, PyObject*) {
                    BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    return iterator->next();
                }
                
                /// __iter__ implementation
                PyObject* tp_iter(PyObject* self) {
                    return self;
                }
                
                /// next() / PyIter_Next() implementation
                PyObject* tp_iternext(PyObject* self) {
                    BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    return iterator->next();
                }
                
                ///////////////////////////////// GETSETTERS /////////////////////////////////
                
                PyObject*    get_length(PyObject* self, void* closure) {
                    BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    Py_ssize_t out = iterator->length();
                    return py::convert(out);
                }
                
                PyObject*    get_remaining(PyObject* self, void* closure) {
                    BatchIterator* iterator = reinterpret_cast<BatchIterator*>(self);
                    Py_ssize_t out = iterator->remaining();
                    return py::convert(out);
                }
                
                namespace methods {
                    
                    PyGetSetDef* getset() {
                        static PyGetSetDef getsets[] = {
                            {
                                (char*)"length",
                                    (getter)py::ext::batch::iterator::get_length,
                                    nullptr,
                                    (char*)"Iterations -> int\n",
                                    nullptr },
                            {
                                (char*)"remaining",
                                    (getter)py::ext::batch::iterator::get_remaining,
                                    nullptr,
                                    (char*)"Remaining iterations -> int\n",
                                    nullptr },
                            { nullptr, nullptr, nullptr, nullptr, nullptr }
                        };
                        return getsets;
                    }
                    
                    PyMethodDef* basic() {
                        static PyMethodDef basics[] = {
                            {
                                "check",
                                    (PyCFunction)py::ext::subtypecheck,
                                    METH_O | METH_CLASS,
                                    "Batch.Iterator.check(putative)\n"
                                    "\t-> Check that an instance is of this type (or a subtype)\n" },
                            {
                                "typecheck",
                                    (PyCFunction)py::ext::typecheck,
                                    METH_O | METH_CLASS,
                                    "Batch.Iterator.typecheck(putative)\n"
                                    "\t-> Check that an instance is strictly an instance of this type\n" },
                            {
                                "next",
                                    (PyCFunction)py::ext::batch::iterator::next,
                                    METH_NOARGS,
                                    "iterator.next()\n"
                                    "\t-> Dereference and advance the iterator \n" },
                            { nullptr, nullptr, 0, nullptr }
                        };
                        return basics;
                    }
                
                }; /* namespace methods */
                
            } /* namespace iterator */
            
            PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                return py::convert(new BatchModel());
            }
            
            int init(PyObject* self, PyObject* args, PyObject* kwargs) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                bool did_extend = false;
                
                /// extend batch with positional arguments
                switch (PyTuple_GET_SIZE(args)) {
                    case 0: return 0;
                    case 1:
                    default: {
                        did_extend = batch->extend(args);
                    }
                }
                
                return did_extend ? 0 : -1;
            }
            
            /// __repr__ implementation
            PyObject* repr(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__repr__();
            }
            
            /// __str__ implementaton -- same as __repr__
            PyObject* str(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__repr__();
            }
            
            /// __hash__ implementation
            long hash(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__hash__();
            }
            
            /// cmp(batch0, batch1) implementaton
            int compare(PyObject* pylhs, PyObject* pyrhs) {
                BatchModel* batch0 = reinterpret_cast<BatchModel*>(pylhs);
                BatchModel* batch1 = reinterpret_cast<BatchModel*>(pyrhs);
                py::ref lhs_compare = batch0->as_pytuple();
                py::ref rhs_compare = batch1->as_pytuple();
                return PyObject_Compare(lhs_compare, rhs_compare);
            }
            
            /// __len__ implementaton
            Py_ssize_t length(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__len__();
            }
            
            /// sq_concat
            PyObject* concat(PyObject* lhs, PyObject* rhs) {
                return py::convert(new BatchModel(lhs, rhs));
            }
            
            /// sq_repeat
            PyObject* repeat(PyObject* basis, Py_ssize_t repeat) {
                return py::convert(new BatchModel(basis, repeat));
            }
            
            /// sq_item
            PyObject* atindex(PyObject* self, Py_ssize_t idx) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return py::object(batch->__index__(idx));
            }
            
            /// sq_slice
            PyObject* atslice(PyObject* basis, Py_ssize_t start, Py_ssize_t end) {
                return py::convert(new BatchModel(basis, start, end));
            }
            
            /// sq_ass_item
            int valueatindex(PyObject* self, Py_ssize_t idx, PyObject* value) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__index__(idx, value) == nullptr ? -1 : 0;
            }
            
            /// sq_contains
            int contains(PyObject* self, PyObject* value) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->contains(value); /// boolean upcast
            }
            
            /// sq_inplace_concat
            PyObject* inplace_concat(PyObject* lhs, PyObject* rhs) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(lhs);
                BatchModel* additional = reinterpret_cast<BatchModel*>(rhs);
                PyObject* list = additional->as_pylist();
                bool did_extend = batch->extend(list);
                return did_extend ? py::object(batch) : nullptr; /// propagate error
            }
            
            /// DEALLOCATE
            void dealloc(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                delete batch;
            }
            
            /// CLEAR
            int clear(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                batch->cleanup(true);
                return 0;
            }
            
            /// TRAVERSE
            int traverse(PyObject* self, visitproc visit, void* arg) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->vacay(visit, arg);
            }
            
            /// __iter__ implementation
            PyObject* tp_iter(PyObject* self) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__iter__();
            }
            
            PyObject* items(PyObject* self, PyObject*) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->as_pylist();
            }
            
            PyObject* iteritems(PyObject* self, PyObject*) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return batch->__iter__();
            }
            
            ///////////////////////////////// LIST API /////////////////////////////////
            
            PyObject* append(PyObject* self, PyObject* obj) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                bool did_append = batch->append(obj);
                return did_append ? py::object(batch) : nullptr;
            }
            
            PyObject* count(PyObject* self, PyObject* obj) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                return py::convert(batch->count(obj));
            }
            
            PyObject* extend(PyObject* self, PyObject* obj) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                bool did_extend = batch->extend(obj);
                return did_extend ? py::object(batch) : nullptr;
            }
            
            PyObject* indexof(PyObject* self, PyObject* args, PyObject* kwargs) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                PyObject* value = nullptr;
                int start = 0;
                int stop = -1;
                char const* keywords[] = { "value", "start", "stop", nullptr };
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "O|ii:index", const_cast<char**>(keywords),
                    &value,             /// "value", PyObject*, thing to look for
                    &start,             /// "start", int, begin() iterator index
                    &stop))             /// "stop", int, end() iterator index
                {
                    return nullptr;
                }
                
                /// will throw ValueErrror if need be:
                Py_ssize_t idx = batch->index(value, start, stop);
                return idx == -1 ? nullptr : py::convert(idx);
            }
            
            PyObject* insert(PyObject* self, PyObject* args, PyObject* kwargs) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                PyObject* obj = nullptr;
                int idx = 0;
                char const* keywords[] = { "index", "object", nullptr };
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "iO:insert", const_cast<char**>(keywords),
                    &idx,        /// "index", int, where to insert
                    &obj))       /// "object", PyObject*, what to insert
                {
                    return nullptr;
                }
                
                bool did_insert = batch->insert(idx, obj);
                return did_insert ? py::object(batch) : nullptr;
            }
            
            PyObject* pop(PyObject* self, PyObject* args, PyObject* kwargs) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                int idx = -1;
                char const* keywords[] = { "index", nullptr };
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|i:pop", const_cast<char**>(keywords),
                    &idx))       /// "index", int, from whence to pop()
                {
                    return nullptr;
                }
                
                /// will throw ValueErrror if need be:
                return batch->pop(idx);
            }
            
            PyObject* removeobj(PyObject* self, PyObject* obj) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                bool did_remove = batch->remove(obj);
                return did_remove ? py::object(batch) : nullptr;
            }
            
            PyObject* reverse(PyObject* self, PyObject*) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                batch->reverse();
                return py::None();
            }
            
            PyObject* sort(PyObject* self, PyObject* args, PyObject* kwargs) {
                using key_tag_t = BatchModel::Tag::KeyCallable;
                using comparison_tag_t = BatchModel::Tag::ComparisonCallable;
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                PyObject* cmp = nullptr;
                PyObject* key = nullptr;
                PyObject* py_reverse = nullptr;
                char const* keywords[] = { "cmp", "key", "reverse", nullptr };
                bool reverse = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|OOO:sort", const_cast<char**>(keywords),
                    &cmp,               /// "cmp", callable PyObject*
                    &key,               /// "key", callable PyObject*
                    &py_reverse))       /// "reverse", boolean PyObject*
                {
                    return nullptr;
                }
                
                reverse = py::options::truth(py_reverse);
                
                if (key != nullptr) {
                    if (cmp == nullptr) {
                        if (reverse) { batch->reverse(); }
                        if (!batch->sort(key, key_tag_t{})) {
                            PyErr_SetString(PyExc_ValueError,
                                "sort(): key sort failed");
                            return nullptr;
                        }
                    } else {
                        PyErr_SetString(PyExc_ValueError,
                            "sort(): use either a key or cmp function (not both)");
                        return nullptr;
                    }
                } else {
                    if (cmp == nullptr) {
                        if (reverse) { batch->reverse(); }
                        if (!batch->sort()) {
                            PyErr_SetString(PyExc_ValueError,
                                "sort(): default sort failed");
                            return nullptr;
                        }
                    } else {
                        if (reverse) { batch->reverse(); }
                        if (!batch->sort(cmp, comparison_tag_t{})) {
                            PyErr_SetString(PyExc_ValueError,
                                "sort(): cmp sort failed");
                            return nullptr;
                        }
                    }
                }
                
                return py::None();
            }
            
            PyObject* write(PyObject* self, PyObject* args, PyObject* kwargs) {
                using iosource_t = typename py::gil::with::source_t;
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                PyObject* py_as_blob = nullptr;
                PyObject* options = nullptr;
                PyObject* file = nullptr;
                Py_buffer view;
                char const* keywords[] = { "destination", "file", "as_blob", "options", nullptr };
                std::string dststr;
                bool as_blob = false,
                     use_file = false,
                     did_save = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|s*OOO:write", const_cast<char**>(keywords),
                    &view,                      /// "destination", buffer with file path
                    &file,                      /// "file", PyFileObject*-castable I/O handle
                    &py_as_blob,                /// "as_blob", Python boolean specifying blobbiness
                    &options))                  /// "options", read-options dict
                {
                    return nullptr;
                }
                
                /// tests are necessary, the next lines choke on nullptr:
                as_blob = py::options::truth(py_as_blob);
                if (file) { use_file = PyFile_Check(file); }
                if (options == nullptr) { options = PyDict_New(); }
                
                if (PyDict_Update(batch->writeoptDict, options) == -1) {
                    Py_DECREF(options);
                    PyErr_SetString(PyExc_SystemError,
                        "Dictionary update failure");
                    return nullptr;
                }
                
                options_map opts = batch->writeopts();
                Py_DECREF(options);
                
                if (as_blob || use_file) {
                    if (!opts.has("format")) {
                        PyErr_SetString(PyExc_AttributeError,
                            "Output format unspecified");
                        return nullptr;
                    }
                }
                
                if (!use_file) {
                    if (as_blob) {
                        py::gil::release nogil;
                        NamedTemporaryFile tf("." + opts.cast<std::string>("format"), false);
                        dststr = std::string(tf.filepath.make_absolute().str());
                    } else {
                        /// save as file -- extract the filename from the buffer
                        py::gil::release nogil;
                        py::buffer::source dest(view);
                        dststr = std::string(dest.str());
                    }
                    if (!dststr.size()) {
                        if (as_blob) {
                            PyErr_SetString(PyExc_ValueError,
                                "Blob output unexpectedly returned zero-length bytestring");
                        } else {
                            PyErr_SetString(PyExc_ValueError,
                                "File output destination path is unexpectedly zero-length");
                        }
                        return nullptr;
                    }
                    did_save = batch->save(dststr.c_str(), opts);
                } else {
                    did_save = batch->savefilelike(file, opts);
                }
                
                if (!did_save) {
                    return nullptr; /// If this is false, PyErr has been set
                }
                
                if (as_blob) {
                    std::vector<byte> data;
                    if (use_file) {
                        py::gil::with iohandle(file);
                        iosource_t readback = iohandle.source();
                        data = readback->full_data();
                    } else {
                        bool removed = false;
                        {
                            py::gil::release nogil;
                            std::unique_ptr<FileSource> readback(
                                new FileSource(dststr.c_str()));
                            data = readback->full_data();
                            readback->close();
                            removed = path::remove(dststr);
                        }
                        if (!removed) {
                            PyErr_Format(PyExc_IOError,
                                "Failed to remove temporary file %s",
                                dststr.c_str());
                            return nullptr;
                        }
                    }
                    return py::string(data);
                }
                /// "else":
                if (use_file) { return py::None(); }
                return py::string(dststr);
            }
            
            ///////////////////////////////// GETSETTERS /////////////////////////////////
            
            PyObject*    get_width(PyObject* self, void* closure) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                Py_ssize_t out = batch->width();
                if (out == -1) { return nullptr; } /// propagate error
                return py::convert(out);
            }
            
            PyObject*    get_height(PyObject* self, void* closure) {
                BatchModel* batch = reinterpret_cast<BatchModel*>(self);
                Py_ssize_t out = batch->height();
                if (out == -1) { return nullptr; } /// propagate error
                return py::convert(out);
            }
            
            namespace methods {
                
                PySequenceMethods* sequence() {
                    static PySequenceMethods sequencemethods = {
                        (lenfunc)py::ext::batch::length,
                        (binaryfunc)py::ext::batch::concat,
                        (ssizeargfunc)py::ext::batch::repeat,
                        (ssizeargfunc)py::ext::batch::atindex,
                        (ssizessizeargfunc)py::ext::batch::atslice, /* ssizessizeargfunc sq_slice */
                        (ssizeobjargproc)py::ext::batch::valueatindex,
                        0, /* ssizessizeobjargproc sq_ass_slice */
                        (objobjproc)py::ext::batch::contains, /* sq_contains */
                        (binaryfunc)py::ext::batch::inplace_concat,
                        0 /* sq_inplace_repeat */
                    };
                    return &sequencemethods;
                }
                
                PyGetSetDef* getset() {
                    static PyGetSetDef getsets[] = {
                        {
                            (char*)"width",
                                (getter)py::ext::batch::get_width,
                                nullptr,
                                (char*)"Batch width -> int\n",
                                nullptr },
                        {
                            (char*)"height",
                                (getter)py::ext::batch::get_height,
                                nullptr,
                                (char*)"Batch height -> int\n",
                                nullptr },
                        { nullptr, nullptr, nullptr, nullptr, nullptr }
                    };
                    return getsets;
                }
                
                PyMethodDef* basic() {
                    static PyMethodDef basics[] = {
                        {
                            "check",
                                (PyCFunction)py::ext::subtypecheck,
                                METH_O | METH_CLASS,
                                "Batch.check(putative)\n"
                                "\t-> Check that an instance is of this type (or a subtype)\n" },
                        {
                            "typecheck",
                                (PyCFunction)py::ext::typecheck,
                                METH_O | METH_CLASS,
                                "Batch.typecheck(putative)\n"
                                "\t-> Check that an instance is strictly an instance of this type\n" },
                        {
                            "items",
                                (PyCFunction)py::ext::batch::items,
                                METH_NOARGS,
                                "batch.items()\n"
                                "\t-> Return a Python list containing the items in the batch \n" },
                        {
                            "iteritems",
                                (PyCFunction)py::ext::batch::iteritems,
                                METH_NOARGS,
                                "batch.iteritems()\n"
                                "\t-> Return a Python iterator over the items in the batch \n" },
                        {
                            "append",
                                (PyCFunction)py::ext::batch::append,
                                METH_O,
                                "batch.append(object)\n"
                                "\t-> Append buffer-compatible object to end\n" },
                        {
                            "count",
                                (PyCFunction)py::ext::batch::count,
                                METH_O,
                                "batch.count(value)\n"
                                "\t-> Return number of occurences of value\n" },
                        {
                            "extend",
                                (PyCFunction)py::ext::batch::extend,
                                METH_O,
                                "batch.extend(iterable)\n"
                                "\t-> Extend by appending buffer-compatible elements from the iterable\n" },
                        {
                            "index",
                                (PyCFunction)py::ext::batch::indexof,
                                METH_VARARGS | METH_KEYWORDS,
                                "batch.index(value, [start, [stop]])\n"
                                "\t-> Return first index of value.\n"
                                "\t   Raises ValueError if the value is not present.\n" },
                        {
                            "insert",
                                (PyCFunction)py::ext::batch::insert,
                                METH_VARARGS | METH_KEYWORDS,
                                "batch.insert(index, object)\n"
                                "\t-> Insert object before index\n" },
                        {
                            "pop",
                                (PyCFunction)py::ext::batch::pop,
                                METH_VARARGS | METH_KEYWORDS,
                                "batch.pop([index])\n"
                                "\t-> Remove and return item at index (default last).\n"
                                "\t   Raises IndexError if list is empty or index is out of range.\n" },
                        {
                            "remove",
                                (PyCFunction)py::ext::batch::removeobj,
                                METH_O,
                                "batch.remove(value)\n"
                                "\t-> Remove first occurrence of value.\n"
                                "\t   Raises ValueError if the value is not present.\n" },
                        {
                            "reverse",
                                (PyCFunction)py::ext::batch::reverse,
                                METH_NOARGS,
                                "batch.reverse()\n"
                                "\t-> Reverse the batch ***IN PLACE*** \n" },
                        {
                            "sort",
                                (PyCFunction)py::ext::batch::sort,
                                METH_VARARGS | METH_KEYWORDS,
                                "batch.sort(cmp=None, key=None, reverse=False)\n"
                                "\t-> Stable sort *IN PLACE* of the batch contents;\n"
                                "\t   cmp(x, y) -> -1, 0, 1\n"
                                "\t   key(item) -> rich-comparable value\n" },
                                {
                            "write",
                                (PyCFunction)py::ext::batch::write,
                                METH_VARARGS | METH_KEYWORDS,
                                "batch.write(destination=\"\", file=None, as_blob=False, options={})\n"
                                "\t-> Format and write image data to file or blob\n"
                                "\t   specifying one of: \n"
                                "\t - a destination file path (destination)\n"
                                "\t - a filehandle opened for writing (file)\n"
                                "\t - a boolean flag requiring data to be returned as bytes (as_blob)\n"
                                "\t   optionally specifying: \n"
                                "\t - format-specific write options (options) \n"
                                "\t   NOTE: \n"
                                "\t - options must contain a 'format' entry, specifying the output format \n"
                                "\t   when write() is called without a destination path. \n" },
                        { nullptr, nullptr, 0, nullptr }
                    };
                    return basics;
                }
                
            }; /* namespace methods */
            
        }; /* namespace batch */
    
    }; /* namespace ext */

}; /* namespace py */


#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BATCHMETHODS_HH_