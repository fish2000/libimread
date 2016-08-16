
#include <cctype>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "flattery.hpp"
#include "detail.hpp"
#include "exceptions.hpp"

namespace py {
    
    namespace flattery {
        
        PyObject* unflatten(PyObject* ignore, PyObject* args) {
            PyObject* src = nullptr;
            PyObject* dst = nullptr;
            PyObject* nonelist = nullptr;
            PyObject* slot = nullptr;
            PyObject* slotvalue = nullptr;
            PyObject* part = nullptr;
            Py_ssize_t pos = 0;
            
            if (!PyArg_ParseTuple(args, "O!:unflatten", &PyDict_Type, &src)) {
                return nullptr;
            }
            
            if (!(dst = PyDict_New())) { goto error; }
            
            /* Create a [None] list. Used for extending lists to higher indices. */
            
            if (!(nonelist = PyList_New(0))) { goto error; }
            if (PyList_Append(nonelist, Py_None) < 0) { goto error; }
            
            /* Iterate through key value pairs in the src dict,
               building the nested data structure in dst as we go. */
            
            PyObject* k;
            PyObject* v;
            
            while (PyDict_Next(src, &pos, &k, &v)) {
                const char* key = PyString_AsString(k);
                const char* p;
                
                p = key;
                slot = dst;
                Py_INCREF(slot);
                
                do {
                    /* Extract current part of the key path. */
                    
                    const char* start = p;
                    while (*p && *p != '.') { p++; }
                    part = PyString_FromStringAndSize(start, p - start);
                    
                    /* Advance to next part of key path, unless at the end. */
                    
                    if (*p == '.') { p++; }
                    
                    /* What value should we insert under this slot?
                       - if this is the last path part, insert the value from src.
                       - if the next path part is numeric, insert an empty list.
                       - otherwise, insert an empty hash.
                     */
                    
                    if (!*p) {
                        slotvalue = v;
                        Py_INCREF(slotvalue);
                    } else if (std::isdigit(*p)) {
                        slotvalue = PyList_New(0);
                    } else {
                        slotvalue = PyDict_New();
                    }
                    
                    if (!slotvalue) { goto error; }
                    
                    /* Assign to the current slot. */
                    
                    if (std::isdigit(*start)) {
                        /* If the current path part is numeric, index into a list. */
                        
                        if (!PyList_Check(slot)) { goto error; }
                        
                        // FIXME thorough error checking here
                        
                        Py_ssize_t len = PyList_Size(slot);
                        Py_ssize_t idx = std::atol(PyString_AsString(part));
                        
                        /* Extend the list with [None,None,...] if necessary. */
                        
                        if (idx >= len) {
                            PyObject* tail = PySequence_Repeat(nonelist, idx - len + 1);
                            PyObject* extended = PySequence_InPlaceConcat(slot, tail);
                            Py_DECREF(tail);
                            Py_DECREF(extended);
                        }
                        
                        /* Don't clobber an existing entry.
                           Caveat: PyList_SetItem() steals a reference to slotvalue. */
                        
                        PyObject* extant = nullptr;
                        
                        if ((extant = PyList_GetItem(slot, idx)) == Py_None) {
                            PyList_SetItem(slot, idx, slotvalue);
                            Py_INCREF(slotvalue);
                        } else {
                            Py_DECREF(slotvalue);
                            slotvalue = extant;
                            Py_INCREF(slotvalue);
                        }
                    } else {
                        /* If the current path part is non-numeric, index into a dict.
                         */
                        
                        if (!PyDict_Check(slot)) { goto error; }
                        
                        /* Don't clobber an existing entry. */
                        
                        PyObject* extant = nullptr;
                        
                        if (!(extant = PyDict_GetItem(slot, part))) {
                            PyDict_SetItem(slot, part, slotvalue);
                        } else {
                            Py_DECREF(slotvalue);
                            slotvalue = extant;
                            Py_INCREF(slotvalue);
                        }
                    }
                    
                    /* Descend further into the dst data structure. */
                    
                    Py_DECREF(slot);
                    slot = slotvalue;
                    slotvalue = nullptr;
                    
                    Py_DECREF(part);
                    part = nullptr;
                    
                } while (*p);
                
                Py_DECREF(slot);
                slot = nullptr;
            }
            
            Py_DECREF(nonelist);
            return dst;
            
            error:
                Py_XDECREF(dst);
                Py_XDECREF(nonelist);
                Py_XDECREF(slot);
                Py_XDECREF(slotvalue);
                Py_XDECREF(part);
            
            return nullptr;
        }
        
        PyObject* flatten(PyObject* input) {
            if (!input) { return py::tuplize(); }
            
            std::vector<py::ref> flattening;
            py::ref flat;
            
            if (PySequence_Check(input)) {
                /// fill vector from sequence
                py::ref sequence = PySequence_Fast(input, "Sequence expected");
                py::ref item(false);
                Py_ssize_t idx = 0,
                        length = PySequence_Fast_GET_SIZE(sequence.get());
                for (; idx < length; ++idx) {
                    item = PySequence_Fast_GET_ITEM(sequence.get(), idx);
                    if (item.none() && idx < length - 1) { continue; }
                    flattening.push_back(py::flattery::flatten(item.get()));
                }
            } else {
                /// SHORTCUT: bump refcount, return input
                return py::object(input);
            }
            
            flat = PyList_New(0);
            
            std::for_each(flattening.begin(),
                          flattening.end(),
                      [&](py::ref const& flattened) {
                /// check again for sequences:
                if (PySequence_Check(flattened.get())) {
                    py::ref sequence = PySequence_Fast(flattened.get(), "Sequence expected");
                    py::ref item(false);
                    Py_ssize_t idx = 0,
                            length = PySequence_Fast_GET_SIZE(sequence.get());
                    for (; idx < length; ++idx) {
                        item = PySequence_Fast_GET_ITEM(sequence.get(), idx);
                        PyList_Append(flat, item);
                    }
                } else {
                    PyList_Append(flat, flattened);
                }
            });
            
            return PyList_AsTuple(flat);
        }
        
        PyObject* flatten_mappings(PyObject* input) {
            py::ref flat;
            py::ref dst;
            py::ref elem(false);
            
            if (PySequence_Check(input)) {
                if (!(flat = PyDict_New())) {
                    return py::SystemError("could not allocate dict");
                }
                py::ref src = PySequence_Fast(input, "Sequence expected");
                
                /* Iterate through elements in the list src, recursively flattening.
                   Skip any entries which are None -- use a sparse encoding. */
                
                Py_ssize_t idx = 0,
                           len = PySequence_Fast_GET_SIZE(src.get());
                
                for (; idx < len; ++idx) {
                    elem = PySequence_Fast_GET_ITEM(src.get(), idx);
                    if (elem.none() && idx < len - 1) {
                        continue;
                    }
                    elem.inc();
                    py::ref o = py::flattery::flatten_mappings(elem);
                    py::ref k = py::format("%zd", idx);
                    py::detail::setitem(flat, k, o.get());
                }
            } else if (PyDict_Check(input)) {
                if (!(flat = PyDict_New())) {
                    return py::SystemError("could not allocate dict");
                }
                
                /* Iterate through pairs in the dict src, recursively flattening. */
                
                py::ref k(false);
                py::ref v(false);
                Py_ssize_t pos = 0;
                
                while (PyDict_Next(input, &pos, &k, &v)) {
                    v.inc();
                    py::ref o = py::flattery::flatten_mappings(v);
                    v.dec();
                    py::detail::setitem(flat, k, o.get());
                }
            } else {
                /* The Python object is a scalar or something we don't know how
                   to flatten, return it as-is. */
                
                return py::object(input);
            }
            
            /* Roll up recursively flattened dictionaries. */
            
            if (!(dst = PyDict_New())) {
                return py::SystemError("could not allocate dict");
            }
            
            py::ref k1(false);
            py::ref v1(false);
            Py_ssize_t pos1 = 0;
            
            while (PyDict_Next(flat, &pos1, &k1, &v1)) {
                if (PyDict_Check(v1)) {
                    py::ref k2(false);
                    py::ref v2(false);
                    Py_ssize_t pos2 = 0;
                    
                    while (PyDict_Next(v1, &pos2, &k2, &v2)) {
                        py::detail::setitemstring(dst,
                            k1.to_string() + "." + k2.to_string(),
                            v2.inc().get());
                    }
                } else {
                    py::detail::setitem(dst, k1, v1.inc().get());
                }
            }
            
            return py::object(dst);
        }
        
        
    }
    
}

