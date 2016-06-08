
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_

#include <cmath>
#include <array>
#include <memory>
#include <string>
#include <iostream>
#include <Python.h>
#include <structmember.h>

#include "private/buffer_t.h"
#include "buffer.hpp"
#include "check.hh"
#include "gil.hpp"
#include "gil-io.hpp"
#include "detail.hpp"
#include "numpy.hpp"
#include "options.hpp"
#include "pybuffer.hpp"
#include "pycapsule.hpp"
#include "typecode.hpp"
#include "hybrid.hh"

#include <libimread/ext/errors/demangle.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/base64.hh>
#include <libimread/errors.hh>
// #include <libimread/memory.hh>
#include <libimread/hashing.hh>
#include <libimread/pixels.hh>

namespace py {
    
    namespace ext {
        
        using im::byte;
        using im::options_map;
        using im::Image;
        using im::ImageFormat;
        using im::HalideNumpyImage;
        using im::ArrayImage;
        using im::HybridFactory;
        using im::ArrayFactory;
        
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
            static char const* typedoc()    { return "Python buffer model base class\n"; }
            
        }; /* BufferModelBase */
        
        template <typename ImageType,
                  typename BufferType = buffer_t,
                  typename FactoryType = typename ImageType::factory_t>
        struct ImageModelBase {
            
            using shared_image_t = std::shared_ptr<ImageType>;
            using weak_image_t = std::weak_ptr<ImageType>;
            
            struct BufferModel : public BufferModelBase<BufferType> {
                
                using base_t = BufferModelBase<BufferType>;
                
                void* operator new(std::size_t newsize) {
                    PyTypeObject* type = FactoryType::buffer_type();
                    return reinterpret_cast<void*>(type->tp_alloc(type, 0));
                }
                
                void operator delete(void* voidself) {
                    BufferModel* self = reinterpret_cast<BufferModel*>(voidself);
                    PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                    if (self->weakrefs != nullptr) { PyObject_ClearWeakRefs(pyself); }
                    self->cleanup();
                    FactoryType::buffer_type()->tp_free(pyself);
                }
                
                weak_image_t image;
                
                BufferModel()
                    :base_t()
                    {}
                
                explicit BufferModel(shared_image_t shared_image)
                    :base_t(shared_image->buffer_ptr())
                    ,image(shared_image)
                    {}
                
                BufferModel(BufferModel const& other)
                    :base_t(other.internal.get())
                    ,image(other.image)
                    {}
                
                /// tag dispatch, reinterpret, depointerize, copy-construct
                explicit BufferModel(PyObject* other)
                    :BufferModel(*reinterpret_cast<BufferModel*>(other))
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
                
                PyObject*   __index__(Py_ssize_t idx, int tc = NPY_UINT8) {
                    Py_ssize_t siz;
                    std::size_t nidx;
                    shared_image_t strong;
                    {
                        py::gil::release nogil;
                        strong = image.lock();
                        siz = strong->size();
                        nidx = static_cast<std::size_t>(idx);
                    }
                    if (siz <= idx || idx < 0) {
                        PyErr_SetString(PyExc_IndexError,
                            "index out of range");
                        return nullptr;
                    }
                    return py::detail::image_typed_idx(strong, tc, nidx);
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
                        literal_pointer = (long)strong->data();
                    }
                    
                    PyObject* map = PyDict_New();
                    imageref_t imageref = *strong.get();
                    py::detail::setitemstring(map, "version",    py::convert(3));
                    py::detail::setitemstring(map, "shape",      py::detail::image_shape(imageref));
                    py::detail::setitemstring(map, "strides",    py::detail::image_strides(imageref));
                    py::detail::setitemstring(map, "descr",      py::detail::structcode_to_dtype(structcode));
                    py::detail::setitemstring(map, "mask",       py::None());
                    py::detail::setitemstring(map, "offset",     py::None());
                    py::detail::setitemstring(map, "data",       py::tuple(PyLong_FromLong(literal_pointer), py::True()));
                    py::detail::setitemstring(map, "typestr",    py::string(dsig));
                    return map;
                }
                
                PyObject* __array_struct__() const {
                    PyArrayInterface* newstruct;
                    {
                        py::gil::release nogil;
                        shared_image_t strong = image.lock();
                        newstruct = py::numpy::array_struct(*strong.get());
                    }
                    
                    return py::cob::objectify<PyArrayInterface, py::cob::single_destructor_t>(
                                              newstruct,
                                              array_destructor<PyArrayInterface>());
                }
            
                static char const* typestring() {
                    static FactoryType factory;
                    static std::string name = "im." + factory.name() + ".Buffer";
                    return name.c_str();
                }
                
                static char const* typedoc()    { return "Python image-backed buffer class\n"; }
                
            }; /* BufferModel */
            
            void* operator new(std::size_t newsize) {
                PyTypeObject* type = FactoryType::image_type();
                return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            }
            
            void operator delete(void* voidself) {
                ImageModelBase* self = reinterpret_cast<ImageModelBase*>(voidself);
                PyObject* pyself = reinterpret_cast<PyObject*>(voidself);
                if (self->weakrefs != nullptr) { PyObject_ClearWeakRefs(pyself); }
                self->cleanup();
                FactoryType::image_type()->tp_free(pyself);
            }
            
            struct Tag {
                struct FromImage            {};
                struct FromBuffer           {};
                struct FromImageBuffer      {};
                struct FromOtherImageBuffer {};
            };
            
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
                :weakrefs(nullptr)
                ,image(other.image)
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(reinterpret_cast<PyObject*>(
                             new typename ImageModelBase::BufferModel(image)))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    Py_INCREF(dtype);
                    Py_INCREF(imagebuffer);
                    Py_INCREF(readoptDict);
                    Py_INCREF(writeoptDict);
                    PyDict_Update(readoptDict,  other.readoptDict);
                    PyDict_Update(writeoptDict, other.writeoptDict);
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
            explicit ImageModelBase(PyObject* other, typename Tag::FromImage tag = typename Tag::FromImage{})
                :ImageModelBase(*reinterpret_cast<ImageModelBase*>(other))
                {}
            
            /// bmoi = Buffer Model Object Instance
            explicit ImageModelBase(BufferModelBase<BufferType> const& bmoi)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(NPY_UINT8, bmoi.internal.get()))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(reinterpret_cast<PyObject*>(
                             new typename ImageModelBase::BufferModel(image)))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                ,clean(false)
                {
                    Py_INCREF(dtype);
                    Py_INCREF(imagebuffer);
                    Py_INCREF(readoptDict);
                    Py_INCREF(writeoptDict);
                }
            
            /// tag dispatch, reinterpret, depointerize, explicit-init-style construct
            explicit ImageModelBase(PyObject* buffer, typename Tag::FromBuffer)
                :ImageModelBase(*reinterpret_cast<BufferModelBase<BufferType>*>(buffer))
                {}
            
            explicit ImageModelBase(int width, int height,
                                    int planes = 1,
                                    int value = 0x00,
                                    int nbits = 8, bool is_signed = false)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(
                       im::detail::for_nbits(nbits, is_signed),
                                             width, height, planes))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(reinterpret_cast<PyObject*>(
                             new typename ImageModelBase::BufferModel(image)))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                ,clean(false)
                {
                    Py_INCREF(dtype);
                    Py_INCREF(imagebuffer);
                    Py_INCREF(readoptDict);
                    Py_INCREF(writeoptDict);
                    if (value > -1) {
                        py::gil::release nogil;
                        std::memset(image->rowp(0), value,
                                    image->size() * dtype->elsize);
                    }
                }
            
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
                FactoryType factory;
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
                    input = std::make_unique<im::FileSource>(source);
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
            
            bool load(Py_buffer const& view, options_map const& opts) {
                py::buffer::source source(view);
                return load(source.str().c_str(), opts);
            }
            
            bool loadfilelike(PyObject* file, options_map const& opts) {
                FactoryType factory;
                std::unique_ptr<ImageFormat> format;
                typename py::gil::with::source_t input;
                std::unique_ptr<Image> output;
                std::string suffix;
                options_map default_opts;
                bool can_read = false;
                
                try {
                    py::gil::with iohandle(file);
                    input = iohandle.source();
                    format = im::for_source(input.get());
                    can_read = format->format_can_read();
                    suffix = format->get_suffix();
                    if (can_read) {
                        default_opts = format->add_options(opts);
                        output = format->read(input.get(), &factory, default_opts);
                        image.reset(dynamic_cast<ImageType*>(output.release()));
                        return true;
                    }
                } catch (im::FormatNotFound& exc) {
                    PyErr_SetString(PyExc_ValueError,
                        "Can't match blob data to a suitable I/O format");
                    return false;
                }
                
                if (format.get()) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented read() in I/O format %s",
                        mime.c_str());
                } else {
                    PyErr_SetString(PyExc_ValueError,
                        "Bad I/O format pointer returned for blob data");
                }
                return false;
            }
            
            bool loadblob(Py_buffer const& view, options_map const& opts) {
                FactoryType factory;
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<py::buffer::source> input;
                std::unique_ptr<Image> output;
                options_map default_opts;
                bool can_read = false;
                
                try {
                    py::gil::release nogil;
                    input = std::make_unique<py::buffer::source>(view);
                    format = im::for_source(input.get());
                    can_read = format->format_can_read();
                    if (can_read) {
                        default_opts = format->add_options(opts);
                        output = format->read(input.get(), &factory, default_opts);
                        image.reset(dynamic_cast<ImageType*>(output.release()));
                        return true;
                    }
                } catch (im::FormatNotFound& exc) {
                    PyErr_SetString(PyExc_ValueError,
                        "Can't match blob data to a suitable I/O format");
                    return false;
                }
                
                if (format.get()) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented read() in I/O format %s",
                        mime.c_str());
                } else {
                    PyErr_SetString(PyExc_ValueError,
                        "Bad I/O format pointer returned for blob data");
                }
                return false;
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
            
            bool savefilelike(PyObject* file, options_map const& opts) {
                std::unique_ptr<ImageFormat> format;
                typename py::gil::with::sink_t output;
                options_map default_opts;
                bool can_write = false;
                
                if (!opts.has("format")) {
                    PyErr_SetString(PyExc_AttributeError,
                        "Output format unspecified in options dict");
                    return false;
                }
                
                try {
                    py::gil::with iohandle(file);
                    output = iohandle.sink();
                    format = im::get_format(
                        opts.cast<char const*>("format"));
                    can_write = format->format_can_write();
                } catch (im::FormatNotFound& exc) {
                    PyErr_Format(PyExc_ValueError,
                        "Can't find I/O format: %s",
                        opts.cast<char const*>("format"));
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
                    default_opts = format->add_options(opts);
                    format->write(dynamic_cast<Image&>(*image.get()),
                                                       output.get(), default_opts);
                }
                
                return true;
            }
            
            PyObject* saveblob(options_map const& opts) {
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<im::FileSink> output;
                std::unique_ptr<im::FileSource> readback;
                std::vector<byte> data;
                std::string ext;
                options_map default_opts;
                bool can_write = false,
                     removed = false,
                     as_url = false,
                     as_html = false;
                
                if (!opts.has("format")) {
                    PyErr_SetString(PyExc_AttributeError,
                        "Output format unspecified in options dict");
                    return nullptr;
                }
                
                try {
                    py::gil::release nogil;
                    ext = opts.cast<std::string>("format");
                    format = im::get_format(ext.c_str());
                    can_write = format->format_can_write();
                } catch (im::FormatNotFound& exc) {
                    PyErr_Format(PyExc_ValueError,
                        "Can't find I/O format: %s",
                        ext.c_str());
                    return nullptr;
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    PyErr_Format(PyExc_ValueError,
                        "Unimplemented write() in I/O format %s",
                        mime.c_str());
                    return nullptr;
                }
                
                {
                    py::gil::release nogil;
                    NamedTemporaryFile tf(format->get_suffix(true),
                                          FILESYSTEM_TEMP_FILENAME,
                                          false);
                    
                    std::string pth = tf.filepath.make_absolute().str();
                    tf.filepath.remove();
                    output = std::make_unique<im::FileSink>(pth.c_str());
                    
                    default_opts = format->add_options(opts);
                    format->write(dynamic_cast<Image&>(*image.get()),
                                                       output.get(), default_opts);
                    output->flush();
                    
                    if (!path::exists(pth)) {
                        py::gil::ensure yesgil;
                        PyErr_SetString(PyExc_ValueError,
                            "Temporary file is AWOL");
                        return nullptr;
                    }
                    
                    readback = std::make_unique<im::FileSource>(pth.c_str());
                    data = readback->full_data();
                    readback->close();
                    readback.reset(nullptr);
                    tf.close();
                    removed = tf.remove();
                }
                
                if (!removed) {
                    PyErr_SetString(PyExc_ValueError,
                        "Failed to remove temporary file");
                    return nullptr;
                }
                
                as_html = opts.cast<bool>("as_html", false);
                as_url = opts.cast<bool>("as_url", as_html);
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
                       Py_TPFLAGS_HAVE_WEAKREFS   |
                       Py_TPFLAGS_HAVE_NEWBUFFER;
            }
            
            static char const* typestring() {
                static FactoryType factory;
                static std::string name = "im." + factory.name();
                return name.c_str();
            }
            
            static char const* typedoc()    { return "Buffered-image multibackend model base class\n"; }
            
        }; /* ImageModelBase */
        
        /// “Models” are python wrapper types
        using ImageModel = ImageModelBase<HalideNumpyImage, buffer_t>;
        using ImageBufferModel = ImageModel::BufferModel;
        
        using ArrayModel = ImageModelBase<ArrayImage, buffer_t>;
        using ArrayBufferModel = ArrayModel::BufferModel;
        
        /// check() has a forward declaration!
        PyObject* check(PyTypeObject* type, PyObject* evaluee);
        
        namespace buffer {
            
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                return reinterpret_cast<PyObject*>(
                    new PythonBufferType());
            }
            
            /// ALLOCATE / frompybuffer(pybuffer_host) implementation
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* newfrompybuffer(PyObject* _nothing_, PyObject* bufferhost) {
                using tag_t = typename PythonBufferType::Tag::FromPyBuffer;
                if (!bufferhost) {
                    PyErr_SetString(PyExc_ValueError,
                        "missing Py_buffer host argument");
                    return nullptr;
                }
                if (!PyObject_CheckBuffer(bufferhost)) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid Py_buffer host");
                    return nullptr;
                }
                return reinterpret_cast<PyObject*>(
                    new PythonBufferType(bufferhost, tag_t{}));
            }
            
            /// ALLOCATE / frombuffer(bufferInstance) implementation
            template <typename BufferType = buffer_t,
                      typename PythonBufferType = BufferModelBase<BufferType>>
            PyObject* newfrombuffer(PyObject* _nothing_, PyObject* buffer) {
                using tag_t = typename PythonBufferType::Tag::FromBuffer;
                if (!buffer) {
                    PyErr_SetString(PyExc_ValueError,
                        "missing im.Buffer argument");
                    return nullptr;
                }
                if (!BufferModel_Check(buffer) &&
                    !ImageBufferModel_Check(buffer) &&
                    !ArrayBufferModel_Check(buffer)) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid im.Buffer instance");
                    return nullptr;
                }
                return reinterpret_cast<PyObject*>(
                    new PythonBufferType(buffer, tag_t{}));
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
                                (PyCFunction)py::ext::check,
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
            (PyCFunction)py::ext::check,
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
                // using tag_t = typename PythonImageType::Tag::FromImage;
                return reinterpret_cast<PyObject*>(
                    new PythonImageType());
            }
            
            /// ALLOCATE / new(width, height, planes, fill, nbits, is_signed) implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* newfromsize(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
                PyObject* py_is_signed = nullptr;
                PyObject* py_fill = nullptr;
                bool is_signed = false;
                uint32_t unbits = 8;
                int width   = -1,
                    height  = -1,
                    planes  = 1,
                    fill    = 0x00,
                    nbits   = 8;
                char const* keywords[] = { "width", "height", "planes",
                                           "fill",  "nbits",  "is_signed", nullptr };
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "ii|iOIO:new", const_cast<char**>(keywords),
                    &width,                 /// "width", int, HOW WIDE <required>
                    &height,                /// "height", int, HOW HIGH <required>
                    &planes,                /// "planes", int, # of channels
                    &py_fill,               /// "fill", Python object providing fill values (int or callable)
                    &unbits,                /// "nbits", unsigned int, # of bits
                    &py_is_signed))         /// "is_signed", Python boolean, value-signedness
                {
                    return nullptr;
                }
                
                is_signed = py::options::truth(py_is_signed);
                
                switch (unbits) {
                    case 1:
                    case 8:
                    case 16:
                    case 32:
                    case 64:
                        nbits = static_cast<int>(unbits);
                        break;
                    default:
                        PyErr_Format(PyExc_ValueError,
                            "bad nbits value: %u (must be 1, 8, 16, 32, or 64)",
                            unbits);
                        return nullptr;
                }
                
                if (py_fill) {
                    if (PyCallable_Check(py_fill)) {
                        /// DO ALL SORTS OF SHIT HERE WITH PyObject_CallObject
                        PyErr_SetString(PyExc_NotImplementedError,
                            "callable filling not implemented");
                        return nullptr;
                    } else {
                        /// We can't use the commented-out line below,
                        /// because 'fill' is currently implemented with
                        /// std::memset() in the constructor we call below.
                        /// Even though std::memset()'s value arg is an int,
                        /// it internally converts it to an unsigned char
                        /// before setting the mem. 
                        if (PyLong_Check(py_fill) || PyInt_Check(py_fill)) {
                            Py_ssize_t fillval = PyInt_AsSsize_t(py_fill);
                            // fill = detail::clamp(fillval, 0L, std::pow(2L, unbits));
                            fill = detail::clamp(fillval, 0L, 255L);
                        }
                    }
                }
                
                /// create new PyObject* image model instance
                PyObject* self = reinterpret_cast<PyObject*>(
                    new PythonImageType(width, height, planes,
                                        fill,  nbits,  is_signed));
                
                /// return the new instance
                return self;
            }
            
            
            /// ALLOCATE / frombuffer(bufferInstance) implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* newfrombuffer(PyObject* _nothing_, PyObject* buffer) {
                using tag_t = typename PythonImageType::Tag::FromBuffer;
                if (!buffer) {
                    PyErr_SetString(PyExc_ValueError,
                        "missing im.Buffer argument");
                    return nullptr;
                }
                if (!BufferModel_Check(buffer) &&
                    !ImageBufferModel_Check(buffer) &&
                    !ArrayBufferModel_Check(buffer)) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid im.Buffer instance");
                    return nullptr;
                }
                return reinterpret_cast<PyObject*>(
                    new PythonImageType(buffer, tag_t{}));
            }
            
            /// ALLOCATE / fromimage(imageInstance) implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* newfromimage(PyObject* _nothing_, PyObject* other) {
                using tag_t = typename PythonImageType::Tag::FromImage;
                if (!other) {
                    PyErr_SetString(PyExc_ValueError,
                        "missing im.Image argument");
                    return nullptr;
                }
                if (!ImageModel_Check(other) &&
                    !ArrayModel_Check(other)) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid im.Image instance");
                    return nullptr;
                }
                return reinterpret_cast<PyObject*>(
                    new PythonImageType(other, tag_t{}));
            }
            
            /// __init__ implementation
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int init(PyObject* self, PyObject* args, PyObject* kwargs) {
                using imagebuffer_t = typename PythonImageType::BufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* py_is_blob = nullptr;
                PyObject* options = nullptr;
                PyObject* file = nullptr;
                Py_buffer view;
                options_map opts;
                char const* keywords[] = { "source", "file", "is_blob", "options", nullptr };
                bool is_blob = false;
                bool did_load = false;
                
                if (!PyArg_ParseTupleAndKeywords(
                    args, kwargs, "|s*OOO:__init__", const_cast<char**>(keywords),
                    &view,                      /// "view", buffer with file path or image data
                    &file,                      /// "file", possible file-like object
                    &py_is_blob,                /// "is_blob", Python boolean specifying blobbiness
                    &options))                  /// "options", read-options dict
                {
                    return -1;
                }
                
                /// test is necessary, the next line chokes on nullptr:
                is_blob = py::options::truth(py_is_blob);
                opts = py::options::parse(options);
                
                if (file) {
                    /// load as file-like Python object
                    did_load = pyim->loadfilelike(file, opts);
                } else if (is_blob) {
                    /// load as blob -- pass the buffer along
                    did_load = pyim->loadblob(view, opts);
                } else {
                    /// load as file -- pass the buffer along
                    did_load = pyim->load(view, opts);
                }
                
                if (!did_load) {
                    /// If this is true, PyErr has already been set
                    PyErr_SetString(PyExc_AttributeError,
                        "Image binary load failed");
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
                static bool named = false;
                static char const* pytypename;
                if (!named) {
                    py::gil::release nogil;
                    pytypename = terminator::nameof(pyim);
                    named = true;
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
                char const* string_ptr;
                Py_ssize_t string_size;
                {
                    py::gil::release nogil;
                    string_ptr = pyim->image->template rowp_as<char const>(0);
                    string_size = pyim->image->size();
                }
                return py::string(string_ptr, string_size);
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
                using imagebuffer_t = typename PythonImageType::BufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->__len__();
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* atindex(PyObject* self, Py_ssize_t idx) {
                using imagebuffer_t = typename PythonImageType::BufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->__index__(idx, static_cast<int>(pyim->dtype->type_num));
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            int getbuffer(PyObject* self, Py_buffer* view, int flags) {
                using imagebuffer_t = typename PythonImageType::BufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                return imbuf->getbuffer(view, flags);
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            void releasebuffer(PyObject* self, Py_buffer* view) {
                using imagebuffer_t = typename PythonImageType::BufferModel;
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
                using iosource_t = typename py::gil::with::source_t;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* py_as_blob = nullptr;
                PyObject* options = nullptr;
                PyObject* file = nullptr;
                Py_buffer view;
                char const* keywords[] = { "destination", "file", "as_blob", "options", nullptr };
                std::string dststr;
                bool as_blob = false;
                bool use_file = false;
                bool did_save = false;
                
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
                
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    Py_DECREF(options);
                    return nullptr;
                }
                
                options_map opts = pyim->writeopts();
                
                if (as_blob || use_file) {
                    if (!opts.has("format")) {
                        PyErr_SetString(PyExc_AttributeError,
                            "Output format unspecified in options dict");
                        return nullptr;
                    }
                }
                
                if (as_blob) {
                    py::gil::release nogil;
                    NamedTemporaryFile tf("." + opts.cast<std::string>("format"),
                                        FILESYSTEM_TEMP_FILENAME, false); /// boolean cleanup on scope exit
                    dststr = std::string(tf.filepath.make_absolute().str());
                } else if (!use_file) {
                    /// save as file -- extract the filename from the buffer
                    py::gil::release nogil;
                    py::buffer::source dest(view);
                    dststr = std::string(dest.str());
                }
                if (!dststr.size() && !use_file) {
                    if (as_blob) {
                        PyErr_SetString(PyExc_ValueError,
                            "Blob output unexpectedly returned zero-length bytestring");
                    } else {
                        PyErr_SetString(PyExc_ValueError,
                            "File output destination path is unexpectedly zero-length");
                    }
                    return nullptr;
                }
                
                if (use_file) {
                    did_save = pyim->savefilelike(file, opts);
                } else {
                    did_save = pyim->save(dststr.c_str(), opts);
                }
                if (!did_save) {
                    return nullptr; /// If this is false, PyErr has been set
                }
                
                if (as_blob) {
                    std::vector<byte> data;
                    bool removed = false;
                    if (use_file) {
                        py::gil::with iohandle(file);
                        iosource_t readback = iohandle.source();
                        data = readback->full_data();
                    } else {
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
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* jupyter_repr_png(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* options = PyDict_New();
                py::detail::setitemstring(options, "format", py::string("png"));
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    Py_DECREF(options);
                    return nullptr;
                }
                return pyim->saveblob(pyim->writeopts());
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* jupyter_repr_jpeg(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* options = PyDict_New();
                py::detail::setitemstring(options, "format", py::string("jpg"));
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    Py_DECREF(options);
                    return nullptr;
                }
                return pyim->saveblob(pyim->writeopts());
            }
            
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject* jupyter_repr_html(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* options = PyDict_New();
                py::detail::setitemstring(options, "format", py::string("jpg"));
                py::detail::setitemstring(options, "as_html", py::True());
                if (PyDict_Update(pyim->writeoptDict, options) == -1) {
                    Py_DECREF(options);
                    return nullptr;
                }
                return pyim->saveblob(pyim->writeopts());
            }
            
            #define DECLARE_CLOSURE(name) static char const* name = #name
            #define CHECK_CLOSURE(name) (char const*)closure == closures::name
            #define BIND_CLOSURE(name) (void*)py::ext::image::closures::name
            
            namespace closures {
                DECLARE_CLOSURE(DTYPE);
                DECLARE_CLOSURE(BUFFER);
                DECLARE_CLOSURE(SHAPE);
                DECLARE_CLOSURE(STRIDES);
                DECLARE_CLOSURE(WIDTH);
                DECLARE_CLOSURE(HEIGHT);
                DECLARE_CLOSURE(PLANES);
                DECLARE_CLOSURE(READ);
                DECLARE_CLOSURE(WRITE);
                DECLARE_CLOSURE(STRUCT);
                DECLARE_CLOSURE(INTERFACE);
            }
            
            /// ImageType.{dtype,buffer} getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_subobject(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                if (CHECK_CLOSURE(DTYPE)) {
                    return py::object(pyim->dtype);
                }
                return py::object(pyim->imagebuffer);
            }
            
            /// ImageType.{shape,strides} getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_liminal_tuple(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                if (CHECK_CLOSURE(STRIDES)) {
                    return py::detail::image_strides(*pyim->image.get());
                }
                return py::detail::image_shape(*pyim->image.get());
            }
            
            /// ImageType.{width,height,planes} getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_dimensional_attribute(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                int idx = CHECK_CLOSURE(WIDTH)  ? 0 : CHECK_CLOSURE(HEIGHT) ? 1 : 2;
                return py::detail::image_dimensional_attribute(*pyim->image.get(), idx);
            }
            
            /// ImageType.{read,write}_opts getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_opts(PyObject* self, void* closure) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                PyObject* target = CHECK_CLOSURE(READ) ? pyim->readoptDict : pyim->writeoptDict;
                return py::object(target);
            }
            
            /// ImageType.{read,write}_opts setter
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
                /// dispatch on closure tag
                if (CHECK_CLOSURE(READ)) {
                    Py_CLEAR(pyim->readoptDict);
                    pyim->readoptDict = py::object(value);
                } else {
                    Py_CLEAR(pyim->writeoptDict);
                    pyim->writeoptDict = py::object(value);
                }
                return 0;
            }
            
            /// ImageType.__array_{struct,interface}__ getter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    get_array_attribute(PyObject* self, void* closure) {
                using imagebuffer_t = typename PythonImageType::BufferModel;
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                imagebuffer_t* imbuf = reinterpret_cast<imagebuffer_t*>(pyim->imagebuffer);
                if (CHECK_CLOSURE(STRUCT)) {
                    return imbuf->__array_struct__();
                }
                return imbuf->__array_interface__();
            }
            
            /// ImageType.read_opts formatter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    format_read_opts(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                std::string out;
                options_map opts = pyim->readopts();
                {
                    py::gil::release nogil;
                    out = opts.format();
                }
                return py::string(out);
            }
            
            /// ImageType.read_opts file-dumper
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    dump_read_opts(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                options_map opts = pyim->readopts();
                return py::options::dump(self, args, kwargs, opts);
            }
            
            /// ImageType.write_opts formatter
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    format_write_opts(PyObject* self, PyObject*) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                std::string out;
                options_map opts = pyim->writeopts();
                {
                    py::gil::release nogil;
                    out = opts.format();
                }
                return py::string(out);
            }
            
            /// ImageType.write_opts file-dumper
            template <typename ImageType = HalideNumpyImage,
                      typename BufferType = buffer_t,
                      typename PythonImageType = ImageModelBase<ImageType, BufferType>>
            PyObject*    dump_write_opts(PyObject* self, PyObject* args, PyObject* kwargs) {
                PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
                options_map opts = pyim->writeopts();
                return py::options::dump(self, args, kwargs, opts);
            }
            
            namespace methods {
                
                template <typename ImageType,
                          typename BufferType = buffer_t>
                PyBufferProcs* buffer() {
                    static PyBufferProcs buffermethods = {
                        0, 0, 0, 0,
                        (getbufferproc)py::ext::image::getbuffer<ImageType, BufferType>,
                        (releasebufferproc)py::ext::image::releasebuffer<ImageType, BufferType>,
                    };
                    return &buffermethods;
                }
                
                template <typename ImageType,
                          typename BufferType = buffer_t>
                PySequenceMethods* sequence() {
                    static PySequenceMethods sequencemethods = {
                        (lenfunc)py::ext::image::length<ImageType, BufferType>,
                        0, 0,
                        (ssizeargfunc)py::ext::image::atindex<ImageType, BufferType>,
                        0, 0, 0, 0
                    };
                    return &sequencemethods;
                }
                
                template <typename ImageType,
                          typename BufferType = buffer_t>
                PyGetSetDef* getset() {
                    static PyGetSetDef getsets[] = {
                        {
                            (char*)"__array_interface__",
                                (getter)py::ext::image::get_array_attribute<ImageType, BufferType>,
                                nullptr,
                                (char*)"NumPy array interface (Python API) -> dict\n",
                                BIND_CLOSURE(INTERFACE) },
                        {
                            (char*)"__array_struct__",
                                (getter)py::ext::image::get_array_attribute<ImageType, BufferType>,
                                nullptr,
                                (char*)"NumPy array struct (C-level API) -> PyCObject\n",
                                BIND_CLOSURE(STRUCT) },
                        {
                            (char*)"dtype",
                                (getter)py::ext::image::get_subobject<ImageType, BufferType>,
                                nullptr,
                                (char*)"Image dtype -> numpy.dtype\n",
                                BIND_CLOSURE(DTYPE) },
                        {
                            (char*)"buffer",
                                (getter)py::ext::image::get_subobject<ImageType, BufferType>,
                                nullptr,
                                (char*)"Underlying data buffer accessor object -> im.Buffer\n",
                                BIND_CLOSURE(BUFFER) },
                        {
                            (char*)"shape",
                                (getter)py::ext::image::get_liminal_tuple<ImageType, BufferType>,
                                nullptr,
                                (char*)"Image shape tuple -> (int, int, int)\n",
                                BIND_CLOSURE(SHAPE) },
                        {
                            (char*)"strides",
                                (getter)py::ext::image::get_liminal_tuple<ImageType, BufferType>,
                                nullptr,
                                (char*)"Image strides tuple -> (int, int, int)\n",
                                BIND_CLOSURE(STRIDES) },
                        {
                            (char*)"width",
                                (getter)py::ext::image::get_dimensional_attribute<ImageType, BufferType>,
                                nullptr,
                                (char*)"Pixel width -> int\n",
                                BIND_CLOSURE(WIDTH) },
                        {
                            (char*)"height",
                                (getter)py::ext::image::get_dimensional_attribute<ImageType, BufferType>,
                                nullptr,
                                (char*)"Pixel height -> int\n",
                                BIND_CLOSURE(HEIGHT) },
                        {
                            (char*)"planes",
                                (getter)py::ext::image::get_dimensional_attribute<ImageType, BufferType>,
                                nullptr,
                                (char*)"Image color planes (channels) -> int\n",
                                BIND_CLOSURE(PLANES) },
                        {
                            (char*)"read_opts",
                                (getter)py::ext::image::get_opts<ImageType, BufferType>,
                                (setter)py::ext::image::set_opts<ImageType, BufferType>,
                                (char*)"Read options -> dict\n",
                                BIND_CLOSURE(READ) },
                        {
                            (char*)"write_opts",
                                (getter)py::ext::image::get_opts<ImageType, BufferType>,
                                (setter)py::ext::image::set_opts<ImageType, BufferType>,
                                (char*)"Write options -> dict\n",
                                BIND_CLOSURE(WRITE) },
                        { nullptr, nullptr, nullptr, nullptr, nullptr }
                    };
                    return getsets;
                }
                
                template <typename ImageType,
                          typename BufferType = buffer_t>
                PyMethodDef* basic() {
                    static PyMethodDef basics[] = {
                        {
                            "_repr_jpeg_",
                                (PyCFunction)py::ext::image::jupyter_repr_jpeg<ImageType, BufferType>,
                                METH_NOARGS,
                                "image._repr_jpeg_()\n"
                                "\t-> Return the image data in the JPEG format\n"
                                "\t-> This method is for use by ipython/jupyter\n" },
                        {
                            "_repr_png_",
                                (PyCFunction)py::ext::image::jupyter_repr_png<ImageType, BufferType>,
                                METH_NOARGS,
                                "image._repr_png_()\n"
                                "\t-> Return the image data in the PNG format\n"
                                "\t-> This method is for use by ipython/jupyter\n" },
                        {
                            "_repr_html_",
                                (PyCFunction)py::ext::image::jupyter_repr_html<ImageType, BufferType>,
                                METH_NOARGS,
                                "image._repr_html_()\n"
                                "\t-> Return the image data as a base64-encoded `data:` URL inside an HTML <img> tag\n"
                                "\t-> This method is for use by ipython/jupyter\n" },
                        {
                            "check",
                                (PyCFunction)py::ext::check,
                                METH_O | METH_CLASS,
                                "ImageType.check(putative)\n"
                                "\t-> Check that an instance is of this type\n" },
                        {
                            "new",
                                (PyCFunction)py::ext::image::newfromsize<ImageType, BufferType>,
                                METH_VARARGS | METH_KEYWORDS | METH_CLASS,
                                "ImageType.new(width, height, planes=1, fill=0x00, nbits=8, is_signed=False)\n"
                                "\t-> Return a new image of size (width, height) \n"
                                "\t   optionally specifying: \n"
                                "\t - number of color channels (planes) \n"
                                "\t - a default fill value (fill) \n"
                                "\t - number of bits per value and/or the signedness (nbits, is_signed)\n" },
                        {
                            "frombuffer",
                                (PyCFunction)py::ext::image::newfrombuffer<ImageType, BufferType>,
                                METH_O | METH_STATIC,
                                "ImageType.frombuffer(buffer)\n"
                                "\t-> Return a new image based on an im.Buffer instance\n" },
                        {
                            "fromimage",
                                (PyCFunction)py::ext::image::newfromimage<ImageType, BufferType>,
                                METH_O | METH_STATIC,
                                "ImageType.fromimage(image)\n"
                                "\t-> Return a new image based on an existing image instance\n" },
                        {
                            "write",
                                (PyCFunction)py::ext::image::write<ImageType, BufferType>,
                                METH_VARARGS | METH_KEYWORDS,
                                "image.write(destination="", file=None, as_blob=False, options={})\n"
                                "\t-> Format and write image data to file or blob\n"
                                "\t   specifying one of: \n"
                                "\t - a destination file path (destination)\n"
                                "\t - a filehandle opened for writing (file)\n"
                                "\t - a boolean flag requiring data to be returned as bytes (as_blob)\n"
                                "\t   optionally specifying: \n"
                                "\t - format-specific write options (options) \n"
                                "\t   NOTE: \n"
                                "\t - options must contain a 'format' entry, specifying the output format \n"
                                "\t   when write() is called without a destination path. \n"
                                 },
                        {
                            "format_read_opts",
                                (PyCFunction)py::ext::image::format_read_opts<ImageType, BufferType>,
                                METH_NOARGS,
                                "image.format_read_opts()\n"
                                "\t-> Get the read options as a formatted JSON string\n" },
                        {
                            "format_write_opts",
                                (PyCFunction)py::ext::image::format_write_opts<ImageType, BufferType>,
                                METH_NOARGS,
                                "image.format_write_opts()\n"
                                "\t-> Get the write options as a formatted JSON string\n" },
                        {
                            "dump_read_opts",
                                (PyCFunction)py::ext::image::dump_read_opts<ImageType, BufferType>,
                                METH_VARARGS | METH_KEYWORDS,
                                "image.dump_read_opts()\n"
                                "\t-> Dump the read options to a JSON file\n" },
                        {
                            "dump_write_opts",
                                (PyCFunction)py::ext::image::dump_write_opts<ImageType, BufferType>,
                                METH_VARARGS | METH_KEYWORDS,
                                "image.dump_write_opts()\n"
                                "\t-> Dump the write options to a JSON file\n" },
                        { nullptr, nullptr, 0, nullptr }
                    };
                    return basics;
                }
                
            } /* namespace methods */
            
        } /* namespace image */
        
    } /* namespace ext */
    
    namespace functions {
        
        PyObject* image_check(PyObject* self, PyObject* args);
        PyObject* buffer_check(PyObject* self, PyObject* args);
        PyObject* imagebuffer_check(PyObject* self, PyObject* args);
        PyObject* array_check(PyObject* self, PyObject* args);
        PyObject* arraybuffer_check(PyObject* self, PyObject* args);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_HALIDEIMAGE_HH_