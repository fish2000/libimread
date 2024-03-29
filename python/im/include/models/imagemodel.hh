
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_IMAGEMODEL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_IMAGEMODEL_HH_

#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <valarray>
// #include <Python.h>
// #include <structmember.h>

#include "base.hh"
#include "../buffer.hpp"
#include "../check.hh"
#include "../gil.hpp"
#include "../gil-io.hpp"
// #include "../detail.hpp"
#include "../exceptions.hpp"
#include "../numpy.hpp"
#include "../options.hpp"
#include "../pybuffer.hpp"
#include "../pycapsule.hpp"
#include "../hybrid.hh"
// #include "base.hh"
#include "buffermodel.hh"

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/base64.hh>
#include <libimread/ext/valarray.hh>
#include <libimread/histogram.hh>
#include <libimread/hashing.hh>

namespace py {
    
    namespace capsule {
        
        using im::Image;
        
        template <typename ContextType> inline
        py::capsule::destructor_t decapsulaton_f() {
            return [](PyObject* capsule) {
                if (PyCapsule_IsValid(capsule, PyCapsule_GetName(capsule))) {
                    char const* name = PyCapsule_GetName(capsule);
                    ContextType* context = (ContextType*)PyCapsule_GetContext(capsule);
                    if (context) { delete context; }
                    if (name) { std::free((void*)name); name = nullptr;    }
                }
            };
        }
        
        template <> inline
        py::capsule::destructor_t decapsulaton_f<PyObject>() {
            return [](PyObject* capsule) {
                if (PyCapsule_IsValid(capsule, PyCapsule_GetName(capsule))) {
                    char const* name = PyCapsule_GetName(capsule);
                    PyObject* context = (PyObject*)PyCapsule_GetContext(capsule);
                    if (context) { Py_DECREF(context); }
                    if (name) { std::free((void*)name); name = nullptr;    }
                }
            };
        }
        
        template <> inline
        py::capsule::destructor_t decapsulaton_f<std::nullptr_t>() {
            return [](PyObject* capsule) {
                if (PyCapsule_IsValid(capsule, PyCapsule_GetName(capsule))) {
                    char const* name = PyCapsule_GetName(capsule);
                    if (name) { std::free((void*)name); name = nullptr;    }
                }
            };
        }
        
    }
    
    namespace ext {
        
        using im::byte;
        using im::Options;
        using im::Image;
        using im::ImageFormat;
        using im::Histogram;
        // using im::HalideNumpyImage;
        using im::ArrayImage;
        // using im::HybridFactory;
        using im::ArrayFactory;
        
        using im::FileSource;
        using im::FileSink;
        using filesystem::path;
        using filesystem::NamedTemporaryFile;
        
        template <typename ImageType,
                  typename BufferType = buffer_t,
                  typename FactoryType = typename ImageType::factory_t>
        struct ImageModelBase : public ModelBase<ImageModelBase<ImageType, BufferType, FactoryType>, true> {
            
            using shared_image_t = std::shared_ptr<ImageType>;
            using weak_image_t = std::weak_ptr<ImageType>;
            
            struct BufferModel : public BufferModelBase<BufferType> {
                
                using base_t = BufferModelBase<BufferType>;
                using Tag = typename BufferModelBase<BufferType>::Tag;
                
                static PyTypeObject* type_ptr() { return FactoryType::buffer_type(); }
                
                // void* operator new(std::size_t newsize) {
                //     PyTypeObject* type = FactoryType::buffer_type();
                //     return reinterpret_cast<void*>(type->tp_alloc(type, 0));
                // }
                //
                // void operator delete(void* voidself) {
                //     BufferModel* self = reinterpret_cast<BufferModel*>(voidself);
                //     PyObject* pyself = py::convert(self);
                //     if (self->weakrefs != nullptr) {
                //         PyObject_ClearWeakRefs(pyself);
                //     }
                //     self->cleanup();
                //     FactoryType::buffer_type()->tp_free(pyself);
                // }
                
                weak_image_t image;
                py::ref beholden;
                
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
                
                explicit BufferModel(ImageModelBase const& imodel)
                    :BufferModel(imodel.image)
                    {}
                
                /// reinterpret, depointerize, copy-construct
                explicit BufferModel(PyObject* other,
                                     typename Tag::FromBuffer)
                    :BufferModel(*reinterpret_cast<BufferModel*>(other))
                    {
                        beholden.set(other);
                    }
                
                /// tag dispatch, reinterpret, depointerize, explicit construct
                explicit BufferModel(PyObject* imodel,
                                     typename Tag::FromImage = typename Tag::FromImage{})
                    :BufferModel(*reinterpret_cast<ImageModelBase*>(imodel))
                    {
                        beholden.set(imodel);
                    }
                
                /// scale constructor only initializes base class -- as such,
                /// it results in a BufferModel instance primarily fit for
                /// reinterpret_cast<BufferModelBase<buffer_t>>-ing:
                explicit BufferModel(PyObject* other,
                                     float scale, int value = 0x00,
                                     typename Tag::ScaledFromBuffer = typename Tag::ScaledFromBuffer{})
                    :base_t(*reinterpret_cast<base_t*>(other), scale, value)
                    {
                        beholden.set(other);
                    }
                
                Py_ssize_t  __len__() {
                    Py_ssize_t out;
                    {
                        py::gil::release nogil;
                        if (shared_image_t strong = image.lock()) {
                            out = strong->size();
                        } else {
                            out = base_t::__len__();
                        }
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
                        return py::IndexError("index out of range");
                    }
                    return py::detail::image_typed_idx(strong.get(), tc, nidx);
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
                
                void cleanup(bool force = false) {
                    base_t::cleanup(force);
                    beholden.clear();
                }
                
                static char const* typestring() {
                    static FactoryType factory;
                    static std::string name = "im." + factory.name() + ".Buffer";
                    return name.c_str();
                }
                
                static char const* typedoc() {
                    return "Python image-backed buffer class\n";
                }
                
            }; /* BufferModel */
            
            static PyTypeObject* type_ptr() { return FactoryType::image_type(); }
            
            // void* operator new(std::size_t newsize) {
            //     PyTypeObject* type = FactoryType::image_type();
            //     return reinterpret_cast<void*>(type->tp_alloc(type, 0));
            // }
            //
            // void operator delete(void* voidself) {
            //     ImageModelBase* self = reinterpret_cast<ImageModelBase*>(voidself);
            //     PyObject* pyself = py::convert(self);
            //     if (self->weakrefs != nullptr) {
            //         PyObject_ClearWeakRefs(pyself);
            //     }
            //     self->cleanup();
            //     FactoryType::image_type()->tp_free(pyself);
            // }
            
            struct Tag {
                struct FromImage            {};
                struct FromBuffer           {};
                struct FromImageBuffer      {};
                struct FromOtherImageBuffer {};
                struct FromImagePlane       {};
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
                ,imagebuffer(py::convert(
                             new typename ImageModelBase::BufferModel(
                                          py::convert(this))))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    PyDict_Update(readoptDict,  other.readoptDict);
                    PyDict_Update(writeoptDict, other.writeoptDict);
                }
            
            /// Plane-copy constructor:
            /// Construct a new ImageModelBase instance from a single plane
            /// of another (as specified by `zidx`)
            explicit ImageModelBase(ImageModelBase const& other, int zidx)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(*other.image.get(), zidx))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(py::convert(
                             new typename ImageModelBase::BufferModel(
                                          py::convert(this))))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    PyDict_Update(readoptDict,  other.readoptDict);
                    PyDict_Update(writeoptDict, other.writeoptDict);
                }
            
            /// Plane-merge constructor:
            /// Construct a new ImageModelBase instance by appending the planes
            /// of the RHS image (`etc`) to those of the LHS (`basis`).
            explicit ImageModelBase(ImageModelBase const& basis,
                                    ImageModelBase const& etc)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(*basis.image.get(),
                                                     *etc.image.get()))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(py::convert(
                             new typename ImageModelBase::BufferModel(
                                          py::convert(this))))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {}
            
            /// Move constructor:
            /// Steal all the resources of the other instance and populate
            /// a new instance with them; mark the other instance as 'clean'
            /// (preventing deallocation)
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
            explicit ImageModelBase(PyObject* other,
                                    typename Tag::FromImage tag = typename Tag::FromImage{})
                :ImageModelBase(*reinterpret_cast<ImageModelBase*>(other))
                {}
            
            /// Take ownership of bmoi pointer --
            /// bmoi = Buffer Model Object Instance
            explicit ImageModelBase(BufferModelBase<BufferType>* bmoi)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(NPY_UINT8, bmoi->internal.get()))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(py::convert(bmoi))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {}
            
            /// tag dispatch, reinterpret, explicit-init-style construct
            explicit ImageModelBase(PyObject* buffer,
                                    typename Tag::FromBuffer)
                :ImageModelBase(reinterpret_cast<BufferModelBase<BufferType>*>(buffer))
                {}
            
            /// tag dispatch, reinterpret, depointerize, explicit-init-style construct
            explicit ImageModelBase(PyObject* source, int zidx,
                                    typename Tag::FromImagePlane = typename Tag::FromImagePlane{})
                :ImageModelBase(*reinterpret_cast<ImageModelBase*>(source), zidx)
                {}
            
            /// Convenience plane-merge forward constructor:
            /// Takes PyObject* instances and forwards the stack values
            /// (q.v. plane-merge constructor supra.) -- NB, this will
            /// Py_DECREF() the LHS (née `basis`); this is so it can be
            /// used on itself in a loop e.g.
            /// 
            ///     pything = new ImageModelBase(pything, addendum);
            /// 
            /// (q.v. py::ext::image::newfrommerge<ImageType, BufferType> sub.)
            /// … sorry if that's awkward, just Py_INCREF() your basis doggie
            explicit ImageModelBase(PyObject* basis, PyObject* etc)
                :ImageModelBase(*reinterpret_cast<ImageModelBase*>(basis),
                                *reinterpret_cast<ImageModelBase*>(etc))
                {
                    Py_DECREF(basis);
                }
            
            /// Value constructor
            /// Coming soon: CALLABLE SUPPORT OH SHIT
            explicit ImageModelBase(int width, int height,
                                    int planes = 1,
                                    int value = 0x00,
                                    int nbits = 8, bool is_signed = false)
                :weakrefs(nullptr)
                ,image(std::make_shared<ImageType>(
                       im::detail::for_nbits(nbits, is_signed),
                                             width, height,
                                             planes > 0 ? planes : 1))
                ,dtype(PyArray_DescrFromType(image->dtype()))
                ,imagebuffer(py::convert(
                             new typename ImageModelBase::BufferModel(
                                          py::convert(this))))
                ,readoptDict(PyDict_New())
                ,writeoptDict(PyDict_New())
                {
                    if (value > -1) {
                        py::gil::release nogil;
                        std::memset(image->rowp(0), value,
                                    image->size() * dtype->elsize);
                    }
                }
            
            void swap(ImageModelBase& other) noexcept {
                using std::swap;
                swap(weakrefs,      other.weakrefs);
                swap(image,         other.image);
                swap(dtype,         other.dtype);
                swap(imagebuffer,   other.imagebuffer);
                swap(readoptDict,   other.readoptDict);
                swap(writeoptDict,  other.writeoptDict);
                swap(clean,         other.clean);
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
                Py_VISIT(imagebuffer);
                Py_VISIT(readoptDict);
                Py_VISIT(writeoptDict);
                return 0;
            }
            
            long __hash__() {
                long out;
                {
                    py::gil::release nogil;
                    auto bithash = blockhash::blockhash(*image);
                    out = static_cast<long>(bithash.to_ulong());
                }
                return out;
            }
            
            std::string modestring() {
                switch (image->planes()) {
                    case 1: return "L";
                    case 2: return "LA";
                    case 3: return "RGB";
                    case 4: return "RGBA";
                    default:
                        std::string out("WAT;");
                        out += std::to_string(image->planes());
                        return out;
                }
            }
            
            PyObject* mode() {
                return py::string(modestring());
            }
            
            PyObject* has_alpha() {
                return py::boolean(!bool(image->planes() % 2));
            }
            
            PyObject* add_alpha() {
                switch (image->planes()) {
                    case 1:
                    case 3: {
                        py::ref alpha = new ImageModelBase(image->dim(0), image->dim(1),
                                                           1, 0xFF,       image->nbits());
                        return py::convert(new ImageModelBase(py::object(this), alpha));
                    }
                    case 2:
                    case 4:
                    default: {
                        return py::AttributeError("Can't add alpha to mode LA/RGBA/WAT image");
                    }
                }
            }
            
            PyObject* remove_alpha() {
                switch (image->planes()) {
                    case 2: {
                        return py::convert(new ImageModelBase(py::convert(this), 0));
                    }
                    case 4:
                    case 5: {
                        ImageModelBase* basis = new ImageModelBase(py::convert(this), 0);
                        int idx = 1,
                            len = image->planes() - 1;
                        for (Py_INCREF(basis); idx < len; ++idx) {
                            py::ref etc = new ImageModelBase(py::convert(this), idx);
                            basis = new ImageModelBase(py::convert(basis), etc);
                        }
                        return py::convert(basis);
                    }
                    case 1:
                    case 3:
                    default: {
                        return py::AttributeError("Can't remove alpha from mode L/RGB/WAT image");
                    }
                }
            }
            
            PyObject* plane_at(int zidx) {
                if (zidx >= image->planes() || zidx < 0) {
                    return py::IndexError("plane_at(): index out of range");
                }
                return py::convert(new ImageModelBase(py::convert(this), zidx));
            }
            
            PyObject* histogram_at(int zidx) {
                if (zidx >= image->planes() || zidx < 0) {
                    return py::IndexError("histogram_at(): index out of range");
                }
                std::valarray<int> inthisto;
                {
                    py::gil::release nogil;
                    Histogram histo(image.get(), zidx);
                    inthisto = valarray::cast<int>(histo.values());
                }
                return py::convert(inthisto);
            }
            
            PyObject* entropy_at(int zidx) {
                if (zidx >= image->planes() || zidx < 0) {
                    return py::IndexError("entropy_at(): index out of range");
                }
                float entropy = 0.0;
                {
                    py::gil::release nogil;
                    Histogram histo(image.get(), zidx);
                    entropy = histo.entropy();
                }
                return py::convert(entropy);
            }
            
            PyObject* otsu_at(int zidx) {
                if (zidx >= image->planes() || zidx < 0) {
                    return py::IndexError("otsu_at(): index out of range");
                }
                int otsu = 0;
                {
                    py::gil::release nogil;
                    Histogram histo(image.get(), zidx);
                    otsu = histo.otsu();
                }
                return py::convert(otsu);
            }
            
            PyObject* histogram_all() {
                std::vector<int> intvec;
                {
                    py::gil::release nogil;
                    int idx = 0,
                        max = image->planes();
                    for (; idx < max; ++idx) {
                        Histogram histo(image.get(), idx);
                        std::transform(histo.begin(), histo.end(),
                                       std::back_inserter(intvec),
                                    [](float v) -> int { return int(v); });
                    }
                }
                return py::convert(intvec);
            }
            
            PyObject* entropy_all() {
                std::vector<float> floatvec;
                {
                    py::gil::release nogil;
                    int idx = 0,
                        max = image->planes();
                    for (; idx < max; ++idx) {
                        Histogram histo(image.get(), idx);
                        floatvec.push_back(histo.entropy());
                    }
                }
                return py::convert(floatvec);
            }
            
            PyObject* otsu_all() {
                std::vector<int> intvec;
                {
                    py::gil::release nogil;
                    int idx = 0,
                        max = image->planes();
                    for (; idx < max; ++idx) {
                        Histogram histo(image.get(), idx);
                        intvec.push_back(histo.otsu());
                    }
                }
                return py::convert(intvec);
            }
            
            PyObject* encapsulate() {
                return py::capsule::encapsulate<ImageType>(image.get(),
                                                           py::object(this), nullptr,
                                                           py::capsule::decapsulaton_f<PyObject>());
            }
            
            PyObject* scale(float scale) {
                using tag_t = typename Tag::FromBuffer;
                using imagebuffer_t = BufferModelBase<BufferType>;
                py::ref scaledbuffer = new imagebuffer_t(imagebuffer, scale);
                ImageModelBase* scaled = new ImageModelBase(scaledbuffer, tag_t{});
                return py::convert(scaled);
            }
            
            Options readopts() {
                return py::options::parse(readoptDict);
            }
            
            Options writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            bool load(char const* source, Options const& opts) {
                FactoryType factory;
                Options default_opts;
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<FileSource> input;
                std::unique_ptr<Image> output;
                bool exists = false,
                     can_read = false;
                
                try {
                    py::gil::release nogil;
                    format = im::for_filename(source);
                    can_read = format->format_can_read();
                    if (can_read) {
                        input = std::make_unique<FileSource>(source);
                        exists = input->exists();
                        if (exists) {
                            default_opts = format->add_options(opts);
                            output = format->read(input.get(), &factory, default_opts);
                            image.reset(dynamic_cast<ImageType*>(output.release()));
                        }
                    }
                    return true;
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError(
                        std::string("Can't find I/O format for file: ") + source,
                        false);
                }
                
                if (!can_read) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(
                        std::string("Unimplemented read() in I/O format ") + mime,
                        false);
                } else if (!exists) {
                    return py::IOError(
                        std::string("Can't find image file: ") + source,
                        false);
                }
                return py::SystemError("Shouldn't have arrived here", false);
            }
            
            bool load(Py_buffer const& view, Options const& opts) {
                py::buffer::source source(view);
                return load(source.str().c_str(), opts);
            }
            
            bool loadfilelike(PyObject* file, Options const& opts) {
                FactoryType factory;
                std::unique_ptr<ImageFormat> format;
                typename py::gil::with::source_t input;
                std::unique_ptr<Image> output;
                Options default_opts;
                bool can_read = false;
                
                try {
                    py::gil::with iohandle(file);
                    input = iohandle.source();
                    format = im::for_source(input.get());
                    can_read = format->format_can_read();
                    if (can_read) {
                        default_opts = format->add_options(opts);
                        output = format->read(input.get(), &factory, default_opts);
                        image.reset(dynamic_cast<ImageType*>(output.release()));
                    }
                    return true;
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError("Can't match blob data to a suitable I/O format", false);
                }
                
                if (format.get()) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(
                        std::string("Unimplemented read() in I/O format ") + mime,
                        false);
                }
                return py::ValueError("Bad I/O format pointer returned for Python file", false);
            }
            
            bool loadblob(Py_buffer const& view, Options const& opts) {
                FactoryType factory;
                std::unique_ptr<ImageFormat> format;
                std::unique_ptr<py::buffer::source> input;
                std::unique_ptr<Image> output;
                Options default_opts;
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
                    return py::ValueError("Can't match blob data to a suitable I/O format", false);
                }
                
                if (format.get()) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(
                        std::string("Unimplemented read() in I/O format ") + mime,
                        false);
                }
                return py::ValueError("Bad I/O format pointer returned for blob data", false);
            }
            
            bool save(char const* destination, Options const& opts) {
                std::unique_ptr<ImageFormat> format;
                Options default_opts;
                bool exists = false,
                     can_write = false,
                     overwrite = true;
                
                try {
                    py::gil::release nogil;
                    format = im::for_filename(destination);
                    can_write = format->format_can_write();
                    exists = path::exists(destination);
                    overwrite = opts.cast<bool>("overwrite", overwrite);
                    if (can_write && exists && overwrite) {
                        path::remove(destination);
                    }
                } catch (im::FormatNotFound& exc) {
                    return py::ValueError(
                        std::string("Can't find I/O format for file: ") + destination,
                        false);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(
                        std::string("Unimplemented write() in I/O format ") + mime,
                        false);
                }
                
                if (exists && !overwrite) {
                    return py::IOError(
                        std::string("File exists (opts['overwrite'] == False): ") + destination,
                        false);
                }
                
                {
                    py::gil::release nogil;
                    std::unique_ptr<FileSink> output(new FileSink(destination));
                    default_opts = format->add_options(opts);
                    format->write(dynamic_cast<Image&>(*image.get()),
                                                       output.get(), default_opts);
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
                
                try {
                    py::gil::with iohandle(file);
                    output = iohandle.sink();
                    ext = opts.cast<std::string>("format");
                    format = im::get_format(ext.c_str());
                    can_write = format->format_can_write();
                    if (can_write) {
                        default_opts = format->add_options(opts);
                        format->write(dynamic_cast<Image&>(*image.get()),
                                                           output.get(), default_opts);
                        return true;
                    }
                } catch (im::FormatNotFound& exc) {
                    return py::IOError(
                        std::string("Can't find I/O format: ") + ext,
                        false);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(
                        std::string("Unimplemented write() in I/O format ") + mime,
                        false);
                }
                
                return py::SystemError("Shouldn't have arrived here", false);
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
                    can_write = format->format_can_write();
                } catch (im::FormatNotFound& exc) {
                    return py::IOError(std::string("Can't find I/O format: ") + ext);
                }
                
                if (!can_write) {
                    std::string mime = format->get_mimetype();
                    return py::IOError(std::string("Unimplemented write() in I/O format ") + mime);
                }
                
                NamedTemporaryFile tf(format->get_suffix(true), false);
                
                {
                    py::gil::release nogil;
                    pth = std::string(tf.filepath.make_absolute().str());
                    tf.filepath.remove();
                    auto output = std::make_unique<FileSink>(pth.c_str());
                    default_opts = format->add_options(opts);
                    format->write(dynamic_cast<Image&>(*image.get()),
                                                       output.get(), default_opts);
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
            
            static char const* typedoc() {
                return "Buffered-image multibackend model base class\n";
            }
            
        }; /* ImageModelBase */
        
    } /* namespace ext */
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_MODELS_IMAGEMODEL_HH_
