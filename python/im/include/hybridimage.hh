
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_HYBRIDIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_HYBRIDIMAGE_HH_

#include <memory>
#include <string>
#include <iostream>
#include <Python.h>
#include <structmember.h>

#include "private/buffer_t.h"
#include "check.hh"
#include "gil.hpp"
#include "detail.hpp"
#include "options.hpp"
#include "pybuffer.hpp"
#include "hybrid.hh"

#include <libimread/libimread.hpp>
#include <libimread/ext/errors/demangle.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/memory.hh>
#include <libimread/hashing.hh>

namespace py {
    
    namespace image {
        
        using im::byte;
        using im::options_map;
        using im::Image;
        using im::ImageFormat;
        using im::HalideNumpyImage;
        using im::HybridFactory;
        
        using filesystem::path;
        using filesystem::NamedTemporaryFile;
        
        template <typename ImageType>
        struct PythonImageBase {
            PyObject_HEAD
            std::unique_ptr<ImageType> image;
            PyArray_Descr* dtype = nullptr;
            PyObject* readoptDict = nullptr;
            PyObject* writeoptDict = nullptr;
            
            options_map readopts() {
                return py::options::parse(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse(writeoptDict);
            }
            
            void cleanup() {
                image.reset(nullptr);
                Py_CLEAR(dtype);
                Py_CLEAR(readoptDict);
                Py_CLEAR(writeoptDict);
            }
        };
        
        /// “Models” are python wrapper types
        using HybridImageModel = PythonImageBase<HalideNumpyImage>;
        
        /// ALLOCATE / __new__ implementation
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* createnew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
            PythonImageType* self = reinterpret_cast<PythonImageType*>(type->tp_alloc(type, 0));
            /// initialize everything to empty values
            if (self != NULL) {
                self->image = std::unique_ptr<ImageType>(nullptr);
                self->dtype = nullptr;
                self->readoptDict = nullptr;
                self->writeoptDict = nullptr;
            }
            return reinterpret_cast<PyObject*>(self); /// all is well, return self
        }
        
        template <typename ImageType = HalideNumpyImage>
        static const std::unique_ptr<ImageType> unique_null_t = std::unique_ptr<ImageType>(nullptr);
        #define UNIQUE_NULL(ImageType) std::unique_ptr<ImageType>(nullptr);
        
        /// Load an instance of the templated image type
        /// from a file source, with specified reading options
        template <typename ImageType = HalideNumpyImage,
                  typename FactoryType = HybridFactory>
        std::unique_ptr<ImageType> load(char const* source,
                                        options_map const& opts) {
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
                return UNIQUE_NULL(ImageType);
            }
            
            if (!can_read) {
                std::string mime = format->get_mimetype();
                PyErr_Format(PyExc_ValueError,
                    "Unimplemented read() in I/O format %s",
                    mime.c_str());
                return UNIQUE_NULL(ImageType);
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
                return UNIQUE_NULL(ImageType);
            }
            
            {
                py::gil::release nogil;
                default_opts = format->add_options(opts);
                output = format->read(input.get(), &factory, default_opts);
                return py::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
        }
        
        /// Load an instance of the templated image type from a Py_buffer
        /// describing an in-memory source, with specified reading options
        template <typename ImageType = HalideNumpyImage,
                  typename FactoryType = HybridFactory>
        std::unique_ptr<ImageType> loadblob(Py_buffer const& view,
                                            options_map const& opts) {
            FactoryType factory;
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
                return UNIQUE_NULL(ImageType);
            }
            
            if (!can_read) {
                std::string mime = format->get_mimetype();
                PyErr_Format(PyExc_ValueError,
                    "Unimplemented read() in I/O format %s",
                    mime.c_str());
                return UNIQUE_NULL(ImageType);
            }
            
            {
                py::gil::release nogil;
                default_opts = format->add_options(opts);
                output = format->read(input.get(), &factory, default_opts);
                return py::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
        }
        
        /// __init__ implementation
        template <typename ImageType = HalideNumpyImage,
                  typename FactoryType = HybridFactory,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int init(PyObject* self, PyObject* args, PyObject* kwargs) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            PyObject* py_is_blob = NULL;
            PyObject* options = NULL;
            Py_buffer view;
            char const* keywords[] = { "source", "is_blob", "options", NULL };
            bool is_blob = false;
            
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
                    pyim->image = py::image::loadblob<ImageType, FactoryType>(view, opts);
                } else {
                    /// load as file -- extract the filename from the buffer
                    /// into a temporary c-string for passing
                    py::buffer::source source(view);
                    std::string srcstr = source.str();
                    char const* srccstr = srcstr.c_str();
                    pyim->image = py::image::load<ImageType, FactoryType>(srccstr, opts);
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
            if (pyim->image == unique_null_t<ImageType>) {
                /// If this is true, PyErr has already been set
                /// ... presumably by problems loading an ImageFormat
                /// or opening the file at the specified image path
                return -1;
            }
            
            /// create and set a dtype based on the loaded image data's type
            Py_CLEAR(pyim->dtype);
            pyim->dtype = PyArray_DescrFromType(pyim->image->dtype());
            Py_INCREF(pyim->dtype);
            
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
                  typename PythonImageType = PythonImageBase<ImageType>>
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
                  typename PythonImageType = PythonImageBase<ImageType>>
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
                  typename PythonImageType = PythonImageBase<ImageType>>
        long hash(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            long out;
            {
                py::gil::release nogil;
                auto bithash = blockhash::blockhash_quick(*pyim->image);
                out = static_cast<long>(bithash.to_ulong());
            }
            return out;
        }
        
        /// __len__ implementation
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        Py_ssize_t length(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            Py_ssize_t out;
            {
                py::gil::release nogil;
                out = pyim->image->size();
            }
            return out;
        }
        
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject* atindex(PyObject* self, Py_ssize_t idx) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            Py_ssize_t siz;
            {
                py::gil::release nogil;
                siz = pyim->image->size();
            }
            if (siz <= idx || idx < 0) {
                PyErr_SetString(PyExc_IndexError,
                    "index out of range");
                return NULL;
            }
            int tc = static_cast<int>(pyim->dtype->type_num);
            std::size_t nidx = static_cast<std::size_t>(idx);
            switch (tc) {
                case NPY_FLOAT: {
                    float op = pyim->image->template rowp_as<float>(0)[nidx];
                    return Py_BuildValue("f", op);
                }
                break;
                case NPY_DOUBLE:
                case NPY_LONGDOUBLE: {
                    double op = pyim->image->template rowp_as<double>(0)[nidx];
                    return Py_BuildValue("d", op);
                }
                break;
                case NPY_USHORT:
                case NPY_UBYTE: {
                    byte op = pyim->image->template rowp_as<byte>(0)[nidx];
                    return Py_BuildValue("B", op);
                }
                break;
                case NPY_UINT: {
                    uint32_t op = pyim->image->template rowp_as<uint32_t>(0)[nidx];
                    return Py_BuildValue("I", op);
                }
                break;
                case NPY_ULONG:
                case NPY_ULONGLONG: {
                    uint64_t op = pyim->image->template rowp_as<uint64_t>(0)[nidx];
                    return Py_BuildValue("Q", op);
                }
                break;
            }
            uint32_t op = pyim->image->template rowp_as<uint32_t>(0)[nidx];
            return Py_BuildValue("I", op);
        }
        
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int getbuffer(PyObject* self, Py_buffer* view, int flags) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            int out;
            {
                py::gil::release nogil;
                out = pyim->image->populate_buffer(view,
                                                   (NPY_TYPES)pyim->dtype->type_num,
                                                   flags);
            }
            return out;
        }
        
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        void releasebuffer(PyObject* self, Py_buffer* view) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            {
                py::gil::release nogil;
                pyim->image->release_buffer(view);
            }
            PyBuffer_Release(view);
        }
        
        /// DEALLOCATE
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        void dealloc(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            pyim->cleanup();
            self->ob_type->tp_free(self);
        }
        
        template <typename ImageType = HalideNumpyImage>
        bool save(ImageType& input, char const* destination,
                                    options_map const& opts) {
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
                format->write(dynamic_cast<Image&>(input), output.get(), default_opts);
            }
            return true;
        }
        
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
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
                did_save = py::image::save<ImageType>(*pyim->image.get(), dststr.c_str(), opts);
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
                out = py::string(data);
                if (out == NULL) {
                    PyErr_SetString(PyExc_ValueError,
                        "Failed converting output to Python string");
                }
                return out;
            }
            /// "else":
            return py::string(dststr);
        }
        
        /// HybridImage.dtype getter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_dtype(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return py::object(pyim->dtype);
        }
        
        /// HybridImage.shape getter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_shape(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return py::tuplize(image->dim(0));
                case 2:
                    return py::tuplize(image->dim(0),
                                       image->dim(1));
                case 3:
                    return py::tuplize(image->dim(0),
                                       image->dim(1),
                                       image->dim(2));
                case 4:
                    return py::tuplize(image->dim(0),
                                       image->dim(1),
                                       image->dim(2),
                                       image->dim(3));
                case 5:
                    return py::tuplize(image->dim(0),
                                       image->dim(1),
                                       image->dim(2),
                                       image->dim(3),
                                       image->dim(4));
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        /// HybridImage.strides getter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_strides(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return py::tuplize(image->stride(0));
                case 2:
                    return py::tuplize(image->stride(0),
                                       image->stride(1));
                case 3:
                    return py::tuplize(image->stride(0),
                                       image->stride(1),
                                       image->stride(2));
                case 4:
                    return py::tuplize(image->stride(0),
                                       image->stride(1),
                                       image->stride(2),
                                       image->stride(3));
                case 5:
                    return py::tuplize(image->stride(0),
                                       image->stride(1),
                                       image->stride(2),
                                       image->stride(3),
                                       image->stride(4));
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        namespace closures {
            static char const* READ  = "READ";
            static char const* WRITE = "WRITE";
        }
        
        /// HybridImage.read_opts getter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_opts(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            PyObject* target = (char const*)closure == closures::READ ? pyim->readoptDict :
                                                                        pyim->writeoptDict;
            return py::object(target);
        }
        
        /// HybridImage.read_opts setter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
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
        
        /// HybridImage.read_opts formatter
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
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
                  typename PythonImageType = PythonImageBase<ImageType>>
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
                  typename PythonImageType = PythonImageBase<ImageType>>
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
        
        /// HybridImage.read_opts file-dumper
        template <typename ImageType = HalideNumpyImage,
                  typename PythonImageType = PythonImageBase<ImageType>>
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
        
} /* namespace py */

using im::HalideNumpyImage;
using im::HybridFactory;
using py::image::HybridImageModel;

static PyBufferProcs HybridImage_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)py::image::getbuffer<HalideNumpyImage>,
    (releasebufferproc)py::image::releasebuffer<HalideNumpyImage>,
};

static PySequenceMethods HybridImage_SequenceMethods = {
    (lenfunc)py::image::length<HalideNumpyImage>,       /* sq_length */
    0,                                                  /* sq_concat */
    0,                                                  /* sq_repeat */
    (ssizeargfunc)py::image::atindex<HalideNumpyImage>, /* sq_item */
    0,                                                  /* sq_slice */
    0,                                                  /* sq_ass_item HAHAHAHA */
    0,                                                  /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                                   /* sq_contains */
};

static PyGetSetDef HybridImage_getset[] = {
    {
        (char*)"dtype",
            (getter)py::image::get_dtype<HalideNumpyImage>,
            nullptr,
            (char*)"HybridImage dtype",
            nullptr },
    {
        (char*)"shape",
            (getter)py::image::get_shape<HalideNumpyImage>,
            nullptr,
            (char*)"HybridImage shape tuple",
            nullptr },
    {
        (char*)"strides",
            (getter)py::image::get_strides<HalideNumpyImage>,
            nullptr,
            (char*)"HybridImage strides tuple",
            nullptr },
    {
        (char*)"read_opts",
            (getter)py::image::get_opts<HalideNumpyImage>,
            (setter)py::image::set_opts<HalideNumpyImage>,
            (char*)"Read options dict",
            (void*)py::image::closures::READ },
    {
        (char*)"write_opts",
            (getter)py::image::get_opts<HalideNumpyImage>,
            (setter)py::image::set_opts<HalideNumpyImage>,
            (char*)"Write options dict",
            (void*)py::image::closures::WRITE },
    { nullptr, nullptr, nullptr, nullptr, nullptr }
};

static PyMethodDef HybridImage_methods[] = {
    {
        "write",
            (PyCFunction)py::image::write<HalideNumpyImage>,
            METH_VARARGS | METH_KEYWORDS,
            "Format and write image data to file or blob" },
    {
        "format_read_opts",
            (PyCFunction)py::image::format_read_opts<HalideNumpyImage>,
            METH_NOARGS,
            "Get the read options as a formatted JSON string" },
    {
        "format_write_opts",
            (PyCFunction)py::image::format_write_opts<HalideNumpyImage>,
            METH_NOARGS,
            "Get the write options as a formatted JSON string" },
    {
        "dump_read_opts",
            (PyCFunction)py::image::dump_read_opts<HalideNumpyImage>,
            METH_VARARGS | METH_KEYWORDS,
            "Dump the read options to a JSON file" },
    {
        "dump_write_opts",
            (PyCFunction)py::image::dump_write_opts<HalideNumpyImage>,
            METH_VARARGS | METH_KEYWORDS,
            "Dump the write options to a JSON file" },
    { nullptr, nullptr, 0, nullptr }
};

static Py_ssize_t HybridImageModel_TypeFlags = Py_TPFLAGS_DEFAULT         | 
                                               Py_TPFLAGS_BASETYPE        | 
                                               Py_TPFLAGS_HAVE_NEWBUFFER;
namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args);
        PyObject* hybridimage_check(PyObject* self, PyObject* args);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_HYBRIDIMAGE_HH_
