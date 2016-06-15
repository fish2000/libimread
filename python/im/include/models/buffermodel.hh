
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BUFFERMODEL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BUFFERMODEL_HH_

#include <cstring>
#include <memory>
#include <string>
#include <Python.h>
#include <structmember.h>

#include "../buffer.hpp"
#include "../check.hh"
#include "../gil.hpp"
#include "../detail.hpp"
#include "../options.hpp"
#include "../numpy.hpp"
#include "base.hh"

#include <libimread/pixels.hh>

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        
        template <typename BufferType = buffer_t>
        struct BufferModelBase : public ModelBase {
            
            using pixel_t = byte;
            using unique_buffer_t = std::unique_ptr<BufferType>;
            using accessor_t = im::pix::accessor<pixel_t>;
            
            static PyTypeObject* type_ptr() { return &BufferModel_Type; }
            
            void* operator new(std::size_t newsize) {
                PyTypeObject* type = type_ptr();
                return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            }
            
            void operator delete(void* voidself) {
                BufferModelBase* self = reinterpret_cast<BufferModelBase*>(voidself);
                PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                if (self->weakrefs != nullptr) { PyObject_ClearWeakRefs(pyself); }
                self->cleanup();
                type_ptr()->tp_free(pyself);
            }
            
            struct Tag {
                struct FromBuffer {};
                struct FromPyBuffer {};
            };
            
            PyObject_HEAD
            PyObject* weakrefs = nullptr;
            bool clean = false;
            unique_buffer_t internal;
            accessor_t accessor;
            
            BufferModelBase()
                :internal(std::make_unique<BufferType>())
                ,accessor{}
                {}
            
            explicit BufferModelBase(BufferType* buffer)
                :clean(true)
                ,internal(unique_buffer_t(buffer))
                ,accessor(internal->host, internal->extent[0] ? internal->stride[0] : 0,
                                          internal->extent[1] ? internal->stride[1] : 0,
                                          internal->extent[2] ? internal->stride[2] : 0)
                {}
            
            BufferModelBase(BufferModelBase const& other)
                :internal(im::buffer::heapcopy(other.internal.get()))
                ,accessor(internal->host, internal->extent[0] ? internal->stride[0] : 0,
                                          internal->extent[1] ? internal->stride[1] : 0,
                                          internal->extent[2] ? internal->stride[2] : 0)
                {}
            
            explicit BufferModelBase(Py_buffer* view)
                :internal(im::buffer::heapcopy(view))
                ,accessor(internal->host, internal->extent[0] ? internal->stride[0] : 0,
                                          internal->extent[1] ? internal->stride[1] : 0,
                                          internal->extent[2] ? internal->stride[2] : 0)
                {}
            
            /// tag dispatch, reinterpret, depointerize, copy-construct
            explicit BufferModelBase(PyObject* other, typename Tag::FromBuffer tag = typename Tag::FromBuffer{})
                :BufferModelBase(*reinterpret_cast<BufferModelBase*>(other))
                {}
            
            explicit BufferModelBase(PyObject* bufferhost, typename Tag::FromPyBuffer) {
                Py_buffer view{ 0 };
                if (PyObject_GetBuffer(bufferhost, &view, PyBUF_ND | PyBUF_STRIDES) != -1) {
                    internal = unique_buffer_t(im::buffer::heapcopy(&view));
                    accessor = accessor_t(internal->host, internal->extent[0] ? internal->stride[0] : 0,
                                                          internal->extent[1] ? internal->stride[1] : 0,
                                                          internal->extent[2] ? internal->stride[2] : 0);
                } else {
                    internal = std::make_unique<BufferType>();
                    accessor = accessor_t{};
                }
                PyBuffer_Release(&view);
            }
            
            void cleanup(bool force = false) {
                if (clean && !force) {
                    internal.release();
                } else {
                    internal.reset(nullptr);
                    clean = !force;
                }
            }
            
            int vacay(visitproc visit, void* arg) { return 0; }
            
            Py_ssize_t __len__() {
                return (internal->extent[0] ? internal->extent[0] : 1) *
                       (internal->extent[1] ? internal->extent[1] : 1) *
                       (internal->extent[2] ? internal->extent[2] : 1) *
                       (internal->extent[3] ? internal->extent[3] : 1);
            }
            
            PyObject* __index__(Py_ssize_t idx, int tc = NPY_UINT8) {
                return py::convert(internal->host[idx]);
            }
            
            PyObject* transpose() {
                std::size_t N, idx;
                {
                    py::gil::release nogil;
                    N = im::buffer::ndims(*internal.get());
                }
                Py_intptr_t permutation[N];
                BufferModelBase* transposed = new BufferModelBase(*this);
                {
                    py::gil::release nogil;
                    for (idx = 0; idx < N; ++idx) {
                        /// can substitute custom permutation mapping, via
                        /// a tuple argument; q.v. numpy array.transpose()
                        permutation[idx] = N - 1 - idx;
                    }
                    for (idx = 0; idx < N; ++idx) {
                        transposed->internal->extent[idx] = internal->extent[permutation[idx]];
                        transposed->internal->stride[idx] = internal->stride[permutation[idx]];
                        transposed->internal->min[idx]    = internal->min[permutation[idx]];
                    }
                }
                return py::convert(transposed);
            }
            
            int getbuffer(Py_buffer* view, int flags, NPY_TYPES dtypenum = NPY_UINT8) {
                {
                    py::gil::release nogil;
                    BufferType* internal_ptr = internal.get();
                    
                    view->buf = internal_ptr->host;
                    view->ndim = 0;
                    view->ndim += internal_ptr->extent[0] ? 1 : 0;
                    view->ndim += internal_ptr->extent[1] ? 1 : 0;
                    view->ndim += internal_ptr->extent[2] ? 1 : 0;
                    view->ndim += internal_ptr->extent[3] ? 1 : 0;
                    
                    view->format = ::strdup(im::detail::structcode(dtypenum));
                    view->shape = new Py_ssize_t[view->ndim];
                    view->strides = new Py_ssize_t[view->ndim];
                    view->itemsize = static_cast<Py_ssize_t>(internal_ptr->elem_size);
                    view->suboffsets = nullptr;
                    
                    int len = 1;
                    for (int idx = 0; idx < view->ndim; idx++) {
                        len *= internal_ptr->extent[idx] ? internal_ptr->extent[idx] : 1;
                        view->shape[idx] = internal_ptr->extent[idx] ? internal_ptr->extent[idx] : 1;
                        view->strides[idx] = internal_ptr->stride[idx] ? internal_ptr->stride[idx] : 1;
                    }
                    
                    view->len = len * view->itemsize;
                    view->readonly = 1; /// true
                    view->internal = (void*)"I HEARD YOU LIKE BUFFERS";
                    view->obj = nullptr;
                }
                return 0;
            }
            
            void releasebuffer(Py_buffer* view) {
                {
                    py::gil::release nogil;
                    if (std::string((const char*)view->internal) == "I HEARD YOU LIKE BUFFERS") {
                        if (view->format)   { std::free(view->format);  view->format  = nullptr; }
                        if (view->shape)    { delete[] view->shape;     view->shape   = nullptr; }
                        if (view->strides)  { delete[] view->strides;   view->strides = nullptr; }
                        view->internal = nullptr;
                    }
                }
                PyBuffer_Release(view);
            }
            
            static constexpr Py_ssize_t typeflags() {
                return Py_TPFLAGS_DEFAULT         |
                       Py_TPFLAGS_BASETYPE        |
                       Py_TPFLAGS_HAVE_GC         |
                       Py_TPFLAGS_HAVE_WEAKREFS   |
                       Py_TPFLAGS_HAVE_NEWBUFFER;
            }
            
            static char const* typestring() { return "im.Buffer"; }
            static char const* typedoc() { 
                return "Python buffer model base class\n";
            }
            
        }; /* BufferModelBase */
        
    } /* namespace ext */
    
} /* namespace py */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_BUFFERMODEL_HH_