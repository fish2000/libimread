
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_

#include <memory>
#include <string>
#include <iostream>
#include <Python.h>
#include <structmember.h>

#include "private/buffer_t.h"
#include "buffer.hpp"
#include "check.hh"
#include "gil.hpp"
#include "detail.hpp"
#include "options.hpp"
#include "preview.hpp"
#include "pybuffer.hpp"
#include "pycapsule.hpp"
#include "typecode.hpp"
#include "hybrid.hh"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/memory.hh>
#include <libimread/hashing.hh>
#include <libimread/pixels.hh>

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        using im::Image;
        using im::ImageFormat;
        using im::HalideNumpyImage;
        using im::HybridFactory;
        
        using filesystem::path;
        using filesystem::NamedTemporaryFile;
        
        template <typename BufferType = buffer_t>
        struct BufferModelBase {
            
            using pixel_t = byte;
            using unique_buffer_t = std::unique_ptr<BufferType>;
            using accessor_t = im::pix::accessor<pixel_t>;
            
            static PyTypeObject* type_ptr() { return &BufferModel_Type; }
            
            void* operator new(std::size_t newsize) {
                PyTypeObject* type = type_ptr();
                BufferModelBase* self = reinterpret_cast<BufferModelBase*>(
                    type->tp_alloc(type, 0));
                if (self != NULL) {
                    self->weakrefs = NULL;
                    self->internal = unique_buffer_t(nullptr);
                }
                return reinterpret_cast<void*>(self);
            }
            
            void operator delete(void* voidself) {
                BufferModelBase* self = reinterpret_cast<BufferModelBase*>(voidself);
                PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                if (self->weakrefs != NULL) { PyObject_ClearWeakRefs(pyself); }
                self->cleanup();
                type_ptr()->tp_free(pyself);
            }
            
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
            
            /// reinterpret, depointerize, copy-construct
            explicit BufferModelBase(PyObject* other)
                :BufferModelBase(*reinterpret_cast<BufferModelBase*>(other))
                {}
            
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
            
            PyObject* __index__(Py_ssize_t idx, int tc = NPY_UINT) {
                return Py_BuildValue("I", internal->host[idx]);
            }
            
            PyObject* transpose() {
                const std::size_t N = im::buffer::ndims(*internal.get());
                
                Py_intptr_t permutation[N];
                for (int idx = 0; idx < N; ++idx) {
                    permutation[idx] = N - 1 - idx;
                }
                
                BufferModelBase* transposed = new BufferModelBase(*this);
                for (int idx = 0; idx < N; ++idx) {
                    transposed->internal->extent[idx] = internal->extent[permutation[idx]];
                    transposed->internal->stride[idx] = internal->stride[permutation[idx]];
                    transposed->internal->min[idx]    = internal->min[permutation[idx]];
                }
                
                return reinterpret_cast<PyObject*>(transposed);
            }
            
            int getbuffer(Py_buffer* view, int flags) {
                {
                    py::gil::release nogil;
                    BufferType* internal_ptr = internal.get();
                    
                    view->buf = internal_ptr->host;
                    view->ndim = 0;
                    view->ndim += internal_ptr->extent[0] ? 1 : 0;
                    view->ndim += internal_ptr->extent[1] ? 1 : 0;
                    view->ndim += internal_ptr->extent[2] ? 1 : 0;
                    view->ndim += internal_ptr->extent[3] ? 1 : 0;
                    
                    view->format = ::strdup(im::detail::structcode(NPY_UINT8));
                    view->shape = new Py_ssize_t[view->ndim];
                    view->strides = new Py_ssize_t[view->ndim];
                    view->itemsize = static_cast<Py_ssize_t>(internal_ptr->elem_size);
                    view->suboffsets = NULL;
                    
                    int len = 1;
                    for (int idx = 0; idx < view->ndim; idx++) {
                        len *= internal_ptr->extent[idx] ? internal_ptr->extent[idx] : 1;
                        view->shape[idx] = internal_ptr->extent[idx] ? internal_ptr->extent[idx] : 1;
                        view->strides[idx] = internal_ptr->stride[idx] ? internal_ptr->stride[idx] : 1;
                    }
                    
                    view->len = len * view->itemsize;
                    view->readonly = 1; /// true
                    view->internal = (void*)"I HEARD YOU LIKE BUFFERS";
                    view->obj = NULL;
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
            
            static char const* typedoc() {
                return "Python buffer model object";
            }
        };
        
        template <typename ImageType,
                  typename BufferType = buffer_t>
        struct ImageModelBase {
            
            using shared_image_t = std::shared_ptr<ImageType>;
            using weak_image_t = std::weak_ptr<ImageType>;
            
            struct ImageBufferModel : public BufferModelBase<BufferType> {
                
                using base_t = BufferModelBase<BufferType>;
                
                static PyTypeObject* type_ptr() { return &ImageBufferModel_Type; }
                
                void* operator new(std::size_t newsize) {
                    PyTypeObject* type = ImageBufferModel::type_ptr();
                    ImageBufferModel* self = reinterpret_cast<ImageBufferModel*>(
                        type->tp_alloc(type, 0));
                    if (self != NULL) {
                        self->weakrefs = NULL;
                        self->internal = typename base_t::unique_buffer_t(nullptr);
                    }
                    return reinterpret_cast<void*>(self);
                }
                
                void operator delete(void* voidself) {
                    ImageBufferModel* self = reinterpret_cast<ImageBufferModel*>(voidself);
                    PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                    if (self->weakrefs != NULL) { PyObject_ClearWeakRefs(pyself); }
                    self->cleanup();
                    ImageBufferModel::type_ptr()->tp_free(pyself);
                }
                
                weak_image_t image;
                
                ImageBufferModel()
                    :base_t()
                    {}
                
                explicit ImageBufferModel(shared_image_t shared_image)
                    :base_t(shared_image->buffer_ptr())
                    ,image(shared_image)
                    {}
                
                ImageBufferModel(ImageBufferModel const& other)
                    :base_t(other.internal.get())
                    ,image(other.image)
                    {}
                
                /// reinterpret, depointerize, copy-construct
                explicit ImageBufferModel(PyObject* other)
                    :ImageBufferModel(*reinterpret_cast<ImageBufferModel*>(other))
                    {}
                
                Py_ssize_t  __len__() {
                    Py_ssize_t out;
                    {
                        py::gil::release nogil;
                        shared_image_t strong = image.lock();
                        out = strong->size();
                    }
                    return out;
                }
                PyObject*   __index__(Py_ssize_t idx, int tc = NPY_UINT) {
                    Py_ssize_t siz;
                    shared_image_t strong;
                    {
                        py::gil::release nogil;
                        strong = image.lock();
                        siz = strong->size();
                    }
                    if (siz <= idx || idx < 0) {
                        PyErr_SetString(PyExc_IndexError,
                            "index out of range");
                        return NULL;
                    }
                    std::size_t nidx = static_cast<std::size_t>(idx);
                    switch (tc) {
                        case NPY_FLOAT: {
                            float op = strong->template rowp_as<float>(0)[nidx];
                            return Py_BuildValue("f", op);
                        }
                        break;
                        case NPY_DOUBLE:
                        case NPY_LONGDOUBLE: {
                            double op = strong->template rowp_as<double>(0)[nidx];
                            return Py_BuildValue("d", op);
                        }
                        break;
                        case NPY_USHORT:
                        case NPY_UBYTE: {
                            byte op = strong->template rowp_as<byte>(0)[nidx];
                            return Py_BuildValue("B", op);
                        }
                        break;
                        case NPY_UINT: {
                            uint32_t op = strong->template rowp_as<uint32_t>(0)[nidx];
                            return Py_BuildValue("I", op);
                        }
                        break;
                        case NPY_ULONG:
                        case NPY_ULONGLONG: {
                            uint64_t op = strong->template rowp_as<uint64_t>(0)[nidx];
                            return Py_BuildValue("Q", op);
                        }
                        break;
                    }
                    uint32_t op = strong->template rowp_as<uint32_t>(0)[nidx];
                    return Py_BuildValue("I", op);
                }
                
                template <typename Pointer = PyArrayInterface,
                          typename = std::nullptr_t>
                py::cob::single_destructor_t array_destructor() const {
                    return [](void* voidptr) {
                        Pointer* pointer = (Pointer*)voidptr;
                        if (pointer) {
                            delete pointer->shape;
                            delete pointer->strides;
                            Py_XDECREF(pointer->descr);
                            delete pointer;
                        }
                    };
                }
                
                PyObject* __array_interface__() const {
                    using imageref_t = typename shared_image_t::element_type const&;
                    shared_image_t strong;
                    char const* structcode;
                    std::string dsig;
                    long literal_pointer;
                    
                    {
                        py::gil::release nogil;
                        strong = image.lock();
                        structcode = strong->structcode();
                        dsig = strong->dsignature();
                        literal_pointer = (long)strong->template rowp_as<uint8_t>(0);
                    }
                    
                    PyObject* map = PyDict_New();
                    imageref_t imageref = *strong.get();
                    PyDict_SetItemString(map, "version",    PyInt_FromSize_t(3));
                    PyDict_SetItemString(map, "shape",      py::detail::image_shape(imageref));
                    PyDict_SetItemString(map, "strides",    py::detail::image_strides(imageref));
                    PyDict_SetItemString(map, "descr",      py::detail::structcode_to_dtype(structcode));
                    PyDict_SetItemString(map, "mask",       py::None());
                    PyDict_SetItemString(map, "offset",     py::None());
                    PyDict_SetItemString(map, "data",       py::tuple(PyLong_FromLong(literal_pointer), py::True()));
                    PyDict_SetItemString(map, "typestr",    py::string(dsig));
                    return map;
                }
                
                PyObject* __array_struct__() const {
                    PyArrayInterface* newstruct;
                    {
                        py::gil::release nogil;
                        shared_image_t strong = image.lock();
                        newstruct = py::detail::array_struct(*strong.get());
                    }
                    
                    return py::cob::objectify<PyArrayInterface, py::cob::single_destructor_t>(
                                              newstruct,
                                              array_destructor<PyArrayInterface>());
                }
                
            }; /* ImageBufferModel */
            
            static PyTypeObject* type_ptr() noexcept { return &ImageModel_Type; }
            
            void* operator new(std::size_t newsize) {
                PyTypeObject* type = ImageModelBase::type_ptr();
                ImageModelBase* self = reinterpret_cast<ImageModelBase*>(
                    type->tp_alloc(type, 0));
                if (self != NULL) {
                    self->weakrefs = NULL;
                    self->image = shared_image_t(nullptr);
                    self->dtype = nullptr;
                    self->imagebuffer = nullptr;
                    self->readoptDict = nullptr;
                    self->writeoptDict = nullptr;
                }
                return reinterpret_cast<void*>(self);
            }
            
            void operator delete(void* voidself) {
                ImageModelBase* self = reinterpret_cast<ImageModelBase*>(voidself);
                PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                if (self->weakrefs != NULL) { PyObject_ClearWeakRefs(pyself); }
                self->cleanup();
                ImageModelBase::type_ptr()->tp_free(pyself);
            }
            
            PyObject_HEAD
            PyObject* weakrefs = nullptr;
            shared_image_t image;
            PyArray_Descr* dtype = nullptr;
            PyObject* imagebuffer = nullptr;
            PyObject* readoptDict = nullptr;
            PyObject* writeoptDict = nullptr;
            bool clean = false;
            
            ImageModelBase()
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>())
                ,dtype(nullptr)
                ,imagebuffer(nullptr)
                ,readoptDict(nullptr)
                ,writeoptDict(nullptr)
                {}
            
            ImageModelBase(ImageModelBase const& other)
                :weakrefs(other.weakrefs)
                ,image(other.image)
                ,dtype(other.dtype)
                ,imagebuffer(other.imagebuffer)
                ,readoptDict(other.readoptDict)
                ,writeoptDict(other.writeoptDict)
                {
                    // Py_INCREF(weakrefs);
                    Py_INCREF(dtype);
                    Py_INCREF(imagebuffer);
                    Py_INCREF(readoptDict);
                    Py_INCREF(writeoptDict);
                }
            
            ImageModelBase(ImageModelBase&& other) noexcept
                :weakrefs(other.weakrefs)
                ,image(std::move(other.image))
                ,dtype(other.dtype)
                ,imagebuffer(other.imagebuffer)
                ,readoptDict(other.readoptDict)
                ,writeoptDict(other.writeoptDict)
                {
                    other.weakrefs = nullptr;
                    other.image.reset(nullptr);
                    other.dtype = nullptr;
                    other.imagebuffer = nullptr;
                    other.readoptDict = nullptr;
                    other.writeoptDict = nullptr;
                    other.clean = true;
                }
            
            /// reinterpret, depointerize, copy-construct
            explicit ImageModelBase(PyObject* other)
                :ImageModelBase(*reinterpret_cast<ImageModelBase*>(other))
                {}
            
            /// bmoi = Buffer Model Object Instance
            explicit ImageModelBase(BufferModelBase<BufferType> const& bmoi)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(NPY_UINT8, bmoi.internal.get()))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(reinterpret_cast<PyObject*>(
                             new typename ImageModelBase::ImageBufferModel(image)))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                ,clean(false)
                {
                    Py_INCREF(dtype);
                    Py_INCREF(imagebuffer);
                    Py_INCREF(readoptDict);
                    Py_INCREF(writeoptDict);
                }
            
            /// tag dipatch, reinterpret, depointerize, explicit-init-style construct
            explicit ImageModelBase(PyObject* buffer, bool buffer_construct_flag = false)
                :ImageModelBase(*reinterpret_cast<BufferModelBase<BufferType>*>(buffer))
                {}
            
            ImageModelBase& operator=(ImageModelBase const& other) {
                if (&other != this) {
                    ImageModelBase(other).swap(*this);
                }
                return *this;
            }
            ImageModelBase& operator=(ImageModelBase&& other) noexcept {
                if (&other != this) {
                    weakrefs = other.weakrefs;
                    image = std::move(other.image);
                    dtype = other.dtype;
                    imagebuffer = other.imagebuffer;
                    readoptDict = other.readoptDict;
                    writeoptDict = other.writeoptDict;
                    other.weakrefs = nullptr;
                    other.image.reset(nullptr);
                    other.dtype = nullptr;
                    other.imagebuffer = nullptr;
                    other.readoptDict = nullptr;
                    other.writeoptDict = nullptr;
                    other.clean = true;
                }
                return *this;
            }
            ImageModelBase& operator=(PyObject* other) {
                if (other != this) {
                    ImageModelBase(*reinterpret_cast<ImageModelBase*>(other)).swap(*this);
                }
                return *this;
            }
            
            void swap(ImageModelBase& other) noexcept {
                using std::swap;
                swap(weakrefs,      other.weakrefs);
                swap(image,         other.image);
                swap(dtype,         other.dtype);
                swap(imagebuffer,   other.imagebuffer);
                swap(readoptDict,   other.readoptDict);
                swap(writeoptDict,  other.writeoptDict);
            }
            
            void cleanup(bool force = false) {
                if (!clean || force) {
                    image.reset();
                    Py_CLEAR(dtype);
                    Py_CLEAR(imagebuffer);
                    Py_CLEAR(readoptDict);
                    Py_CLEAR(writeoptDict);
                    clean = !force;
                }
            }
            
            /// Arguments to ImageModel::vacay() are as required
            /// by the Py_VISIT(), w/r/t both types and names
            int vacay(visitproc visit, void* arg) {
                Py_VISIT(dtype);
                Py_VISIT(readoptDict);
                Py_VISIT(writeoptDict);
                return 0;
            }
            
            long __hash__() {
                long out;
                {
                    py::gil::release nogil;
                    auto bithash = blockhash::blockhash_quick(*image);
                    out = static_cast<long>(bithash.to_ulong());
                }
                return out;
            }
            
            options_map readopts() {
                return py::options::parse(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            bool load(char const* source, options_map const& opts) {
                HybridFactory factory;
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<im::FileSource> input;
                std::unique_ptr<Image> output;
                bool exists = false,
                     can_read = false;
                options_map default_opts;
                
                try {
                    py::gil::release nogil;
                    format = im::for_filename(source);
                    can_read = format->format_can_read();
                } catch (im::FormatNotFound& exc) {
                    PyErr_Format(PyExc_ValueError,
                        "Can't find I/O format for file: %s", source);
                    return false;
                }
                
                if (!can_read) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented read() in I/O format %s",
                        mime.c_str());
                    return false;
                }
                
                {
                    py::gil::release nogil;
                    input = std::unique_ptr<im::FileSource>(
                        new im::FileSource(source));
                    exists = input->exists();
                }
                
                if (!exists) {
                    PyErr_Format(PyExc_ValueError,
                        "Can't find image file: %s", source);
                    return false;
                }
                
                {
                    py::gil::release nogil;
                    default_opts = format->add_options(opts);
                    output = format->read(input.get(), &factory, default_opts);
                    image.reset(dynamic_cast<ImageType*>(output.release()));
                }
                
                return true;
            }
            
            bool loadblob(Py_buffer const& view, options_map const& opts) {
                HybridFactory factory;
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<py::buffer::source> input;
                std::unique_ptr<Image> output;
                options_map default_opts;
                bool can_read = false;
                
                try {
                    py::gil::release nogil;
                    input = std::unique_ptr<py::buffer::source>(
                        new py::buffer::source(view));
                    format = im::for_source(input.get());
                    can_read = format->format_can_read();
                } catch (im::FormatNotFound& exc) {
                    PyErr_SetString(PyExc_ValueError,
                        "Can't match blob data to a suitable I/O format");
                    return false;
                }
                
                if (!can_read) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented read() in I/O format %s",
                        mime.c_str());
                    return false;
                }
                
                {
                    py::gil::release nogil;
                    default_opts = format->add_options(opts);
                    output = format->read(input.get(), &factory, default_opts);
                    image.reset(dynamic_cast<ImageType*>(output.release()));
                }
                
                return true;
            }
            
            bool save(char const* destination, options_map const& opts) {
                std::unique_ptr<ImageFormat> format;
                options_map default_opts;
                bool exists = false,
                     can_write = false,
                     overwrite = true;
                
                try {
                    py::gil::release nogil;
                    format = im::for_filename(destination);
                    can_write = format->format_can_write();
                } catch (im::FormatNotFound& exc) {
                    PyErr_Format(PyExc_ValueError,
                        "Can't find I/O format for file: %s", destination);
                    return false;
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented write() in I/O format %s",
                        mime.c_str());
                    return false;
                }
                
                {
                    py::gil::release nogil;
                    exists = path::exists(destination);
                    overwrite = opts.cast<bool>("overwrite", true);
                }
                
                if (exists && !overwrite) {
                    PyErr_Format(PyExc_ValueError,
                        "File exists (opts['overwrite'] == False): %s",
                        destination);
                    return false;
                }
                {
                    py::gil::release nogil;
                    if (exists && overwrite) {
                        path::remove(destination);
                    }
                    std::unique_ptr<im::FileSink> output(new im::FileSink(destination));
                    default_opts = format->add_options(opts);
                    format->write(dynamic_cast<Image&>(*image.get()),
                                                       output.get(), default_opts);
                }
                
                return true;
            }
            
            static constexpr Py_ssize_t typeflags() {
                return Py_TPFLAGS_DEFAULT         |
                       Py_TPFLAGS_BASETYPE        |
                       Py_TPFLAGS_HAVE_GC         |
                       Py_TPFLAGS_HAVE_WEAKREFS   |
                       Py_TPFLAGS_HAVE_NEWBUFFER;
            }
            
            static char const* typedoc() {
                return "Python image model object";
            }
            
        }; /* ImageModelBase */
        
        /// “Models” are python wrapper types
        using BufferModel = BufferModelBase<buffer_t>;
        using ImageModel = ImageModelBase<HalideNumpyImage, buffer_t>;
        using ImageBufferModel = ImageModel::ImageBufferModel;
        
        namespace buffer {
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                return reinterpret_cast<PyObject*>(new PythonBufferType());
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
                char const* pytypename;
                {
                    py::gil::release nogil;
                    pytypename = terminator::nameof(pybuf);
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
                Py_ssize_t string_size = pybuf->__len__();
                return PyString_FromStringAndSize(
                    (char const*)pybuf->internal->host,
                    string_size);
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
                return py::object(pybuf->transpose());
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
            PyObject*    get_array_interface(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return py::object(pybuf->__array_interface__());
            }
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject*    get_array_struct(PyObject* self, void* closure) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                return py::object(pybuf->__array_struct__());
            }
            
            /// tostring() -- like __str__ implementation (above), return bytes from buffer
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* tostring(PyObject* self, PyObject*) {
                PythonBufferType* pybuf = reinterpret_cast<PythonBufferType*>(self);
                Py_ssize_t string_size = pybuf->__len__();
                return PyString_FromStringAndSize(
                    (char const*)pybuf->internal->host,
                    string_size);
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
                pybuf->vacay(visit, arg);
                return 0;
            }
            
        }; /* namespace buffer */
    
    }; /* namespace ext */

}; /* namespace py */

using py::ext::BufferModel;

static PyBufferProcs Buffer_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
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
            (char*)"Buffer with transposed axes",
            nullptr },
    {
        (char*)"shape",
            (getter)py::ext::buffer::get_shape<buffer_t>,
            nullptr,
            (char*)"Buffer shape tuple",
            nullptr },
    {
        (char*)"strides",
            (getter)py::ext::buffer::get_strides<buffer_t>,
            nullptr,
            (char*)"Buffer strides tuple",
            nullptr },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyMethodDef Buffer_methods[] = {
    {
        "tostring",
            (PyCFunction)py::ext::buffer::tostring<buffer_t>,
            METH_NOARGS,
            "Get bytes from buffer as a string" },
    {
        "transpose",
            (PyCFunction)py::ext::buffer::transpose<buffer_t>,
            METH_NOARGS,
            "Get copy of buffer with transposed axes" },
    { nullptr, nullptr, 0, nullptr }
};

using py::ext::BufferModel;
using py::ext::ImageBufferModel;

static PyBufferProcs ImageBuffer_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)py::ext::buffer::getbuffer<buffer_t, ImageBufferModel>,
    (releasebufferproc)py::ext::buffer::releasebuffer<buffer_t, ImageBufferModel>,
};

static PySequenceMethods ImageBuffer_SequenceMethods = {
    (lenfunc)py::ext::buffer::length<buffer_t, ImageBufferModel>,       /* sq_length */
    0,                                                                  /* sq_concat */
    0,                                                                  /* sq_repeat */
    (ssizeargfunc)py::ext::buffer::atindex<buffer_t, ImageBufferModel>, /* sq_item */
    0,                                                                  /* sq_slice */
    0,                                                                  /* sq_ass_item HAHAHAHA */
    0,                                                                  /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                                                   /* sq_contains */
};

static PyGetSetDef ImageBuffer_getset[] = {
    {
        (char*)"__array_interface__",
            (getter)py::ext::buffer::get_array_interface<buffer_t, ImageBufferModel>,
            nullptr,
            (char*)"NumPy array interface (Python API)",
            nullptr },
    {
        (char*)"__array_struct__",
            (getter)py::ext::buffer::get_array_struct<buffer_t, ImageBufferModel>,
            nullptr,
            (char*)"NumPy array interface (C-level API)",
            nullptr },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyMethodDef ImageBuffer_methods[] = {
    {
        "tostring",
            (PyCFunction)py::ext::buffer::tostring<buffer_t, ImageBufferModel>,
            METH_NOARGS,
            "Get bytes from image buffer as a string" },
    { nullptr, nullptr, 0, nullptr }
};

namespace py {
    
    namespace ext {
        
        namespace image {
            
            /// ALLOCATE / __new__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                return reinterpret_cast<PyObject*>(
                    new PythonImageType());
            }
            
            /// ALLOCATE / frombuffer(bufferInstance) implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* newfrombuffer(PyObject* _nothing_, PyObject* buffer) {
                if (!buffer) {
                    PyErr_SetString(PyExc_ValueError,
                        "missing im.Buffer argument");
                    return NULL;
                }
                if (!BufferModel_Check(buffer) &&
                    !ImageBufferModel_Check(buffer)) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid im.Buffer instance");
                    return NULL;
                }
                return reinterpret_cast<PyObject*>(
                    new PythonImageType(buffer, true));
            }
            
            /// __init__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int init(PyObject* self, PyObject* args, PyObject* kwargs) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* py_is_blob = NULL;
                PyObject* options = NULL;
                Py_buffer view;
                char const* keywords[] = { "source", "is_blob", "options", NULL };
                bool is_blob = false;
                bool did_load = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "s*|OO", const_cast<char**>(keywords),
                    &view,                      /// "view", buffer with file path or image data
                    &py_is_blob,                /// "is_blob", Python boolean specifying blobbiness
                    &options))                  /// "options", read-options dict
                {
                    return -1;
                }
                
                /// test is necessary, the next line chokes on NULL:
                is_blob = py::options::truth(py_is_blob);
                
                try {
                    options_map opts = py::options::parse(options);
                    if (is_blob) {
                        /// load as blob -- pass the buffer along
                        did_load = pyim->loadblob(view, opts);
                    } else {
                        /// load as file -- extract the filename from the buffer
                        /// into a temporary c-string for passing
                        py::buffer::source source(view);
                        std::string srcstr = source.str();
                        char const* srccstr = srcstr.c_str();
                        did_load = pyim->load(srccstr, opts);
                    }
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return -1;
                } catch (im::NotImplementedError& exc) {
                    /// this shouldn't happen -- a generic ImageFormat pointer
                    /// was returned when determining the blob image type
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return -1;
                }
                if (!did_load) {
                    /// If this is true, PyErr has already been set
                    /// ... presumably by problems loading an ImageFormat
                    /// or opening the file at the specified image path
                    return -1;
                }
                
                /// create and set a dtype based on the loaded image data's type
                Py_CLEAR(pyim->dtype);
                pyim->dtype = PyArray_DescrFromType(pyim->image->dtype());
                Py_INCREF(pyim->dtype);
                
                /// allocate a new image buffer
                Py_CLEAR(pyim->imagebuffer);
                pyim->imagebuffer = reinterpret_cast<PyObject*>(new imagebuffer_t(pyim->image));
                Py_INCREF(pyim->imagebuffer);
                
                /// store the read options dict
                Py_CLEAR(pyim->readoptDict);
                pyim->readoptDict = options ? options : PyDict_New();
                Py_INCREF(pyim->readoptDict);
                
                /// ... and now OK, store an empty write options dict
                Py_CLEAR(pyim->writeoptDict);
                pyim->writeoptDict = PyDict_New();
                Py_INCREF(pyim->writeoptDict);
                
                /// ALL IS WELL:
                return 0;
            }
            
            /// __repr__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* repr(PyObject* self) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                char const* pytypename;
                {
                    py::gil::release nogil;
                    pytypename = terminator::nameof(pyim);
                }
                return PyString_FromFormat(
                    "< %s @ %p >",
                    pytypename, pyim);
            }
            
            /// __str__ implementaton -- return bytes of image
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* str(PyObject* self) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                Py_ssize_t string_size;
                {
                    py::gil::release nogil;
                    string_size = pyim->image->size();
                }
                return PyString_FromStringAndSize(
                    pyim->image->template rowp_as<char const>(0),
                    string_size);
            }
            
            /// __hash__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            long hash(PyObject* self) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                return pyim->__hash__();
            }
            
            /// __len__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            Py_ssize_t length(PyObject* self) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->__len__();
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* atindex(PyObject* self, Py_ssize_t idx) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->__index__(idx, static_cast<int>(pyim->dtype->type_num));
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int getbuffer(PyObject* self, Py_buffer* view, int flags) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->getbuffer(view, flags);
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            void releasebuffer(PyObject* self, Py_buffer* view) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                imbuf->releasebuffer(view);
            }
            
            /// DEALLOCATE
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            void dealloc(PyObject* self) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                delete pyim;
            }
            
            /// CLEAR
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int clear(PyObject* self) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                pyim->cleanup(true);
                return 0;
            }
            
            /// TRAVERSE
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int traverse(PyObject* self, visitproc visit, void* arg) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                pyim->vacay(visit, arg);
                return 0;
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* write(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* py_as_blob = NULL;
                PyObject* options = NULL;
                Py_buffer view;
                char const* keywords[] = { "destination", "as_blob", "options", NULL };
                std::string dststr;
                bool as_blob = false;
                bool did_save = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|s*OO", const_cast<char**>(keywords),
                    &view,                      /// "destination", buffer with file path
                    &py_as_blob,                /// "as_blob", Python boolean specifying blobbiness
                    &options))                  /// "options", read-options dict
                {
                    return NULL;
                }
                
                /// tests are necessary, the next lines choke on NULL:
                as_blob = py::options::truth(py_as_blob);
                if (options == NULL) { options = PyDict_New(); }
                
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    /// some exception was raised somewhere
                    return NULL;
                }
                
                try {
                    options_map opts = pyim->writeopts();
                    
                    if (as_blob) {
                        /// save as blob -- pass the buffer along
                        if (!opts.has("format")) {
                            PyErr_SetString(PyExc_AttributeError,
                                "Output format unspecified in options dict");
                            return NULL;
                        }
                        {
                            py::gil::release nogil;
                            NamedTemporaryFile tf("." + opts.cast<std::string>("format"),  /// suffix
                                                  FILESYSTEM_TEMP_FILENAME,                /// prefix (filename template)
                                                  false);                                  /// cleanup on scope exit
                            dststr = std::string(tf.filepath.make_absolute().str());
                        }
                    } else {
                        /// save as file -- extract the filename from the buffer
                        /// into a temporary string for passing
                        py::gil::release nogil;
                        py::buffer::source dest(view);
                        dststr = std::string(dest.str());
                    }
                    did_save = pyim->save(dststr.c_str(), opts);
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                } catch (im::NotImplementedError& exc) {
                    /// this shouldn't happen -- a generic ImageFormat pointer
                    /// was returned when determining the blob image type
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                }
                
                if (!did_save) {
                    /// If this is false, PyErr has already been set
                    /// ... presumably by problems loading an ImageFormat
                    /// or opening the file at the specified image path
                    return NULL;
                }
                
                if (!dststr.size()) {
                    if (as_blob) {
                        PyErr_SetString(PyExc_ValueError,
                            "Blob output unexpectedly returned zero-length bytestring");
                        return NULL;
                    } else {
                        PyErr_SetString(PyExc_ValueError,
                            "File output destination path is unexpectedly zero-length");
                        return NULL;
                    }
                }
                
                if (as_blob) {
                    std::vector<byte> data;
                    PyObject* out = NULL;
                    bool removed = false;
                    {
                        py::gil::release nogil;
                        std::unique_ptr<im::FileSource> readback(
                            new im::FileSource(dststr.c_str()));
                        data = readback->full_data();
                        readback->close();
                        readback.reset(nullptr);
                        removed = path::remove(dststr);
                    }
                    if (!removed) {
                        PyErr_Format(PyExc_ValueError,
                            "Failed to remove temporary file %s",
                            dststr.c_str());
                        return NULL;
                    }
                    out = PyString_FromStringAndSize(
                        (char const*)&data[0], data.size());
                    if (out == NULL) {
                        PyErr_SetString(PyExc_ValueError,
                            "Failed converting output to Python string");
                    }
                    return out;
                }
                /// "else":
                return py::string(dststr);
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* preview(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* options = NULL;
                options_map opts;
                char const* keywords[] = { "options", NULL };
                std::string dststr;
                bool did_save = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|O", const_cast<char**>(keywords),
                    &options))                  /// "options", read-options dict
                {
                    return py::False();
                }
                if (options == NULL) { options = PyDict_New(); }
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    return py::False();
                }
                
                try {
                    opts = pyim->writeopts();
                    if (!opts.has("format")) {
                        PyErr_SetString(PyExc_AttributeError,
                            "Output format unspecified in options dict");
                        return py::False();
                    }
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return py::False();
                }
                
                try {
                    py::gil::release nogil;
                    NamedTemporaryFile tf("." + opts.cast<std::string>("format"),  /// suffix
                                        FILESYSTEM_TEMP_FILENAME,                  /// prefix (filename template)
                                        false);                                    /// cleanup on scope exit
                    path dst = tf.filepath.make_absolute();
                    dststr = std::string(dst.str());
                    did_save = pyim->save(dststr.c_str(), opts);
                    if (!did_save) { return NULL; }
                    py::image::preview(dst);
                    tf.close();
                    tf.remove();
                } catch (im::NotImplementedError& exc) {
                    /// this shouldn't happen
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return py::False();
                }
                
                return py::True();
            }
            
            /// HybridImage.dtype getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_dtype(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                return py::object(pyim->dtype);
            }
            
            /// HybridImage.imagebuffer getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_imagebuffer(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                return py::object(pyim->imagebuffer);
            }
            
            /// HybridImage.shape getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_shape(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                return py::detail::image_shape(*pyim->image.get());
            }
            
            /// HybridImage.strides getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_strides(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                return py::detail::image_strides(*pyim->image.get());
            }
            
            namespace closures {
                static char const* READ  = "READ";
                static char const* WRITE = "WRITE";
            }
            
            /// HybridImage.{read,write}_opts getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_opts(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* target = (char const*)closure == closures::READ ? pyim->readoptDict :
                                                                            pyim->writeoptDict;
                return py::object(target);
            }
            
            /// HybridImage.{read,write}_opts setter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int          set_opts(PyObject* self, PyObject* value, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                if (!value) { return 0; }
                if (!PyMapping_Check(value)) {
                    PyErr_SetString(PyExc_AttributeError,
                        "opts value must be dict-ish");
                    return -1;
                }
                /// switch on closure tag (?)
                if ((char const*)closure == closures::READ) {
                    Py_CLEAR(pyim->readoptDict);
                    pyim->readoptDict = py::object(value);
                } else {
                    Py_CLEAR(pyim->writeoptDict);
                    pyim->writeoptDict = py::object(value);
                }
                return 0;
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_array_interface(PyObject* self, void* closure) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return py::object(imbuf->__array_interface__());
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_array_struct(PyObject* self, void* closure) {
                using imagebuffer_t = typename PythonImageType::ImageBufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return py::object(imbuf->__array_struct__());
            }
            
            /// HybridImage.read_opts formatter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    format_read_opts(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                std::string out;
                options_map opts;
                
                try {
                    opts = pyim->readopts();
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                }
                
                {
                    py::gil::release nogil;
                    out = opts.format();
                }
                
                return py::string(out);
            }
            
            /// HybridImage.read_opts file-dumper
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    dump_read_opts(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                options_map opts;
                try {
                    opts = pyim->readopts();
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                }
                return py::options::dump(self, args, kwargs, opts);
            }
            
            /// HybridImage.write_opts formatter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    format_write_opts(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                std::string out;
                options_map opts;
                
                try {
                    opts = pyim->writeopts();
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                }
                
                {
                    py::gil::release nogil;
                    out = opts.format();
                }
                
                return py::string(out);
            }
            
            /// HybridImage.write_opts file-dumper
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    dump_write_opts(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                options_map opts;
                try {
                    opts = pyim->writeopts();
                } catch (im::OptionsError& exc) {
                    PyErr_SetString(PyExc_AttributeError, exc.what());
                    return NULL;
                }
                return py::options::dump(self, args, kwargs, opts);
            }
        
        } /* namespace image */
        
    } /* namespace ext */
        
} /* namespace py */

using im::HalideNumpyImage;
using im::HybridFactory;
using py::ext::ImageModel;

static PyBufferProcs Image_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)py::ext::image::getbuffer<HalideNumpyImage, buffer_t>,
    (releasebufferproc)py::ext::image::releasebuffer<HalideNumpyImage, buffer_t>,
};

static PySequenceMethods Image_SequenceMethods = {
    (lenfunc)py::ext::image::length<HalideNumpyImage, buffer_t>,         /* sq_length */
    0,                                                                      /* sq_concat */
    0,                                                                      /* sq_repeat */
    (ssizeargfunc)py::ext::image::atindex<HalideNumpyImage, buffer_t>,   /* sq_item */
    0,                                                                      /* sq_slice */
    0,                                                                      /* sq_ass_item HAHAHAHA */
    0,                                                                      /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                                                       /* sq_contains */
};

static PyGetSetDef Image_getset[] = {
    {
        (char*)"__array_interface__",
            (getter)py::ext::image::get_array_interface<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"NumPy array interface (Python API)",
            nullptr },
    {
        (char*)"__array_struct__",
            (getter)py::ext::image::get_array_struct<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"NumPy array interface (C-level API)",
            nullptr },
    {
        (char*)"dtype",
            (getter)py::ext::image::get_dtype<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"Image dtype",
            nullptr },
    {
        (char*)"buffer",
            (getter)py::ext::image::get_imagebuffer<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"Underlying data buffer accessor object",
            nullptr },
    {
        (char*)"shape",
            (getter)py::ext::image::get_shape<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"Image shape tuple",
            nullptr },
    {
        (char*)"strides",
            (getter)py::ext::image::get_strides<HalideNumpyImage, buffer_t>,
            nullptr,
            (char*)"Image strides tuple",
            nullptr },
    {
        (char*)"read_opts",
            (getter)py::ext::image::get_opts<HalideNumpyImage, buffer_t>,
            (setter)py::ext::image::set_opts<HalideNumpyImage, buffer_t>,
            (char*)"Read options dict",
            (void*)py::ext::image::closures::READ },
    {
        (char*)"write_opts",
            (getter)py::ext::image::get_opts<HalideNumpyImage, buffer_t>,
            (setter)py::ext::image::set_opts<HalideNumpyImage, buffer_t>,
            (char*)"Write options dict",
            (void*)py::ext::image::closures::WRITE },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyMethodDef Image_methods[] = {
    {
        "frombuffer",
            (PyCFunction)py::ext::image::newfrombuffer<HalideNumpyImage, buffer_t>,
            METH_O | METH_STATIC,
            "Return a new im.Image based on an im.Buffer instance" },
    {
        "write",
            (PyCFunction)py::ext::image::write<HalideNumpyImage, buffer_t>,
            METH_VARARGS | METH_KEYWORDS,
            "Format and write image data to file or blob" },
    {
        "preview",
            (PyCFunction)py::ext::image::preview<HalideNumpyImage, buffer_t>,
            METH_VARARGS | METH_KEYWORDS,
            "Preview image in external viewer" },
    {
        "format_read_opts",
            (PyCFunction)py::ext::image::format_read_opts<HalideNumpyImage, buffer_t>,
            METH_NOARGS,
            "Get the read options as a formatted JSON string" },
    {
        "format_write_opts",
            (PyCFunction)py::ext::image::format_write_opts<HalideNumpyImage, buffer_t>,
            METH_NOARGS,
            "Get the write options as a formatted JSON string" },
    {
        "dump_read_opts",
            (PyCFunction)py::ext::image::dump_read_opts<HalideNumpyImage, buffer_t>,
            METH_VARARGS | METH_KEYWORDS,
            "Dump the read options to a JSON file" },
    {
        "dump_write_opts",
            (PyCFunction)py::ext::image::dump_write_opts<HalideNumpyImage, buffer_t>,
            METH_VARARGS | METH_KEYWORDS,
            "Dump the write options to a JSON file" },
    { nullptr, nullptr, 0, nullptr }
};

namespace py {
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args);
        PyObject* buffer_check(PyObject* self, PyObject* args);
        PyObject* imagebuffer_check(PyObject* self, PyObject* args);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_
