
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BUFFERMETHODS_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BUFFERMETHODS_HH_

#include <memory>
#include <string>
#include <Python.h>
#include <structmember.h>

#include "../buffer.hpp"
#include "../check.hh"
#include "../gil.hpp"
#include "../detail.hpp"
#include "../exceptions.hpp"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/pixels.hh>

#include "typecheck.hh"
#include "../models/buffermodel.hh"

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        
        namespace buffer {
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                return py::convert(new PythonBufferType());
            }
            
            /// ALLOCATE / frompybuffer(pybuffer_host) implementation
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* newfrompybuffer(PyObject* _nothing_, PyObject* bufferhost) {
                using tag_t = typename PythonBufferType::Tag::FromPyBuffer;
                if (!bufferhost) {
                    return py::ValueError("missing Py_buffer host argument");
                }
                if (!PyObject_CheckBuffer(bufferhost)) {
                    return py::ValueError("invalid Py_buffer host");
                }
                return py::convert(new PythonBufferType(bufferhost, tag_t{}));
            }
            
            /// ALLOCATE / frombuffer(bufferInstance) implementation
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* newfrombuffer(PyObject* _nothing_, PyObject* buffer) {
                using tag_t = typename PythonBufferType::Tag::FromBuffer;
                if (!buffer) {
                    return py::ValueError("missing im.Buffer argument");
                }
                if (!BufferModel_Check(buffer) &&
                    !ImageBufferModel_Check(buffer) &&
                    !ArrayBufferModel_Check(buffer)) {
                    return py::ValueError("invalid im.Buffer instance");
                }
                return py::convert(new PythonBufferType(buffer, tag_t{}));
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            int init(PyObject* self, PyObject* args, PyObject* kwargs) {
                // PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return 0;
            }
            
            /// __repr__ implementation
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* repr(PyObject* self) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                static bool named = false;
                static char const* pytypename;
                if (!named) {
                    py::gil::release nogil;
                    pytypename = terminator::nameof(pybuf);
                    named = true;
                }
                return PyString_FromFormat(
                    "< %s @ %p >",
                    pytypename, pybuf);
            }
            
            /// __str__ implementaton -- return bytes from buffer
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* str(PyObject* self) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                std::size_t string_size = static_cast<std::size_t>(pybuf->__len__());
                return py::string((char const*)pybuf->internal->host, string_size);
            }
            
            /// cmp(buffer0, buffer1) implementaton (uses __str__)
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            int compare(PyObject* pylhs, PyObject* pyrhs) {
                py::ref lhs_compare = PyObject_Str(pylhs);
                py::ref rhs_compare = PyObject_Str(pyrhs);
                return PyObject_Compare(lhs_compare, rhs_compare);
            }
            
            /// __len__ implementaton
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            Py_ssize_t length(PyObject* self) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->__len__();
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* atindex(PyObject* self, Py_ssize_t idx) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->__index__(idx);
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            int getbuffer(PyObject* self, Py_buffer* view, int flags) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->getbuffer(view, flags);
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            void releasebuffer(PyObject* self, Py_buffer* view) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                pybuf->releasebuffer(view);
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    transpose(PyObject* self, PyObject*) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->transpose();
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_transpose(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->transpose();
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_shape(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return im::buffer::shape<BufferType>(*pybuf->internal.get());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_strides(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return im::buffer::strides<BufferType>(*pybuf->internal.get());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_width(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return im::buffer::width<BufferType>(*pybuf->internal.get());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_height(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return im::buffer::height<BufferType>(*pybuf->internal.get());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_planes(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return im::buffer::planes<BufferType>(*pybuf->internal.get());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_array_interface(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->__array_interface__();
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_array_struct(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->__array_struct__();
            }
            
            /// tostring() -- like __str__ implementation (above), return bytes from buffer
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* tostring(PyObject* self, PyObject*) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                std::size_t string_size = static_cast<std::size_t>(pybuf->__len__());
                return py::string((char const*)pybuf->internal->host, string_size);
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* scale(PyObject* self, PyObject* scale_factor) {
                // PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                float factor = (float)PyFloat_AsDouble(scale_factor);
                return py::convert(new PythonBufferType(self, factor));
            }
            
            /// DEALLOCATE
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            void dealloc(PyObject* self) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                delete pybuf;
            }
            
            /// CLEAR
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            int clear(PyObject* self) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                pybuf->cleanup(true);
                return 0;
            }
            
            /// TRAVERSE
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            int traverse(PyObject* self, visitproc visit, void* arg) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return pybuf->vacay(visit, arg);
            }
            
            namespace methods {
                
                template <typename BufferType = buffer_t,
                          typename PythonBufferType = BufferModelBase<BufferType>>
                PyBufferProcs* buffer() {
                    static PyBufferProcs buffermethods = {
                        0, 0, 0, 0,
                        (getbufferproc)py::ext::buffer::getbuffer<BufferType, PythonBufferType>,
                        (releasebufferproc)py::ext::buffer::releasebuffer<BufferType, PythonBufferType>,
                    };
                    return &buffermethods;
                }
                
                template <typename BufferType = buffer_t,
                          typename PythonBufferType = BufferModelBase<BufferType>>
                PySequenceMethods* sequence() {
                    static PySequenceMethods sequencemethods = {
                        (lenfunc)py::ext::buffer::length<BufferType, PythonBufferType>,
                        0, 0,
                        (ssizeargfunc)py::ext::buffer::atindex<BufferType, PythonBufferType>,
                        0, 0, 0, 0
                    };
                    return &sequencemethods;
                }
                
                template <typename BufferType = buffer_t,
                          typename PythonBufferType = BufferModelBase<BufferType>>
                PyGetSetDef* getset() {
                    static PyGetSetDef getsets[] = {
                        {
                            (char*)"__array_interface__",
                                (getter)py::ext::buffer::get_array_interface<BufferType, PythonBufferType>,
                                nullptr,
                                (char*)"NumPy array interface (Python API) -> dict\n",
                                nullptr },
                        {
                            (char*)"__array_struct__",
                                (getter)py::ext::buffer::get_array_struct<BufferType, PythonBufferType>,
                                nullptr,
                                (char*)"NumPy array interface (C-level API) -> PyCObject\n",
                                nullptr },
                        { nullptr, nullptr, nullptr, nullptr, nullptr }
                    };
                    return getsets;
                }
                
                template <typename BufferType = buffer_t,
                          typename PythonBufferType = BufferModelBase<BufferType>>
                PyMethodDef* basic() {
                    static PyMethodDef basics[] = {
                        {
                            "check",
                                (PyCFunction)py::ext::typecheck,
                                METH_O | METH_CLASS,
                                "BufferType.check(putative)\n"
                                "\t-> Check the type of an instance against BufferType\n" },
                        {
                            "tobytes",
                                (PyCFunction)py::ext::buffer::tostring<BufferType, PythonBufferType>,
                                METH_NOARGS,
                                "buffer.tobytes()\n"
                                "\t-> Get bytes from image buffer\n" },
                        {
                            "tostring",
                                (PyCFunction)py::ext::buffer::tostring<BufferType, PythonBufferType>,
                                METH_NOARGS,
                                "buffer.tostring()\n"
                                "\t-> Get bytes from image buffer (buffer.tobytes() alias)\n" },
                        {
                            "scale",
                                (PyCFunction)py::ext::buffer::scale<BufferType, PythonBufferType>,
                                METH_O,
                                "buffer.scale(factor)\n"
                                "\t-> Return a scaled copy of the buffer\n" },
                        { nullptr, nullptr, 0, nullptr }
                    };
                    return basics;
                }
                
            }; /* namespace methods */
            
        }; /* namespace buffer */
    
    }; /* namespace ext */

}; /* namespace py */

static PyBufferProcs Buffer_Buffer3000Methods = {
    0, 0, 0, 0,
    (getbufferproc)py::ext::buffer::getbuffer<buffer_t>,
    (releasebufferproc)py::ext::buffer::releasebuffer<buffer_t>,
};

static PySequenceMethods Buffer_SequenceMethods = {
    (lenfunc)py::ext::buffer::length<buffer_t>,         /* sq_length */
    0,                                                  /* sq_concat */
    0,                                                  /* sq_repeat */
    (ssizeargfunc)py::ext::buffer::atindex<buffer_t>,   /* sq_item */
    0,                                                  /* sq_slice */
    0,                                                  /* sq_ass_item HAHAHAHA */
    0,                                                  /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                                   /* sq_contains */
};

static PyGetSetDef Buffer_getset[] = {
    {
        (char*)"T",
            (getter)py::ext::buffer::get_transpose<buffer_t>,
            nullptr,
            (char*)"Transpose of buffer array data (per buffer.transpose()) -> im.Buffer\n",
            nullptr },
    {
        (char*)"shape",
            (getter)py::ext::buffer::get_shape<buffer_t>,
            nullptr,
            (char*)"Buffer shape -> (int, int, int)\n",
            nullptr },
    {
        (char*)"strides",
            (getter)py::ext::buffer::get_strides<buffer_t>,
            nullptr,
            (char*)"Buffer strides -> (int, int, int)\n",
            nullptr },
    {
        (char*)"width",
            (getter)py::ext::buffer::get_width<buffer_t>,
            nullptr,
            (char*)"Buffer width -> int\n",
            nullptr },
    {
        (char*)"height",
            (getter)py::ext::buffer::get_height<buffer_t>,
            nullptr,
            (char*)"Buffer height -> int\n",
            nullptr },
    {
        (char*)"planes",
            (getter)py::ext::buffer::get_planes<buffer_t>,
            nullptr,
            (char*)"Buffer color planes -> int\n",
            nullptr },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyMethodDef Buffer_methods[] = {
    {
        "check",
            (PyCFunction)py::ext::typecheck,
            METH_O | METH_CLASS,
            "im.Buffer.check(putative)\n"
            "\t-> Check the type of an instance against im.Buffer\n" },
    {
        "frombuffer",
            (PyCFunction)py::ext::buffer::newfrombuffer<buffer_t>,
            METH_O | METH_STATIC,
            "im.Buffer.frombuffer(buffer)\n"
            "\t-> Return a new im.Buffer based on a buffer_t host object\n" },
    {
        "frompybuffer",
            (PyCFunction)py::ext::buffer::newfrompybuffer<buffer_t>,
            METH_O | METH_STATIC,
            "im.Buffer.frombuffer(pybuffer_host)\n"
            "\t-> Return a new im.Buffer based on a Py_buffer host object\n" },
    {
        "tobytes",
            (PyCFunction)py::ext::buffer::tostring<buffer_t>,
            METH_NOARGS,
            "buffer.tobytes()\n"
            "\t-> Get bytes from buffer\n" },
    {
        "tostring",
            (PyCFunction)py::ext::buffer::tostring<buffer_t>,
            METH_NOARGS,
            "buffer.tostring()\n"
            "\t-> Get bytes from buffer (buffer.tobytes() alias)\n" },
    {
        "transpose",
            (PyCFunction)py::ext::buffer::transpose<buffer_t>,
            METH_NOARGS,
            "buffer.transpose()\n"
            "\t-> Get a transpose of the image array\n"
            "\t   SEE ALSO:\n"
            "\t - buffer.T (property)\n"
            "\t - numpy.array.transpose() and numpy.array.T\n" },
    {
        "scale",
            (PyCFunction)py::ext::buffer::scale<buffer_t>,
            METH_O,
            "buffer.scale(factor)\n"
            "\t-> Return a scaled copy of the buffer\n" },
    { nullptr, nullptr, 0, nullptr }
};

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_PYMETHODS_BUFFERMETHODS_HH_