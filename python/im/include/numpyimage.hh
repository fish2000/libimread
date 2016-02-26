
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_

#include <memory>
#include <string>
#include <Python.h>
#include <structmember.h>

#include "gil.hh"
#include "detail.hh"
#include "options.hpp"
#include "pybuffer.hpp"
#include "numpy.hh"

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
        using im::HybridArray;
        using im::ArrayFactory;
        
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
                return py::options::parse_options(readoptDict);
            }
            
            options_map writeopts() {
                return py::options::parse_options(writeoptDict);
            }
            
            void cleanup() {
                image.reset(nullptr);
                Py_CLEAR(dtype);
                Py_CLEAR(readoptDict);
                Py_CLEAR(writeoptDict);
            }
            
            ~PythonImageBase() { cleanup(); }
        };
        
        using NumpyImage = PythonImageBase<HybridArray>;
        
        /// ALLOCATE / __new__ implementation
        template <typename ImageType = HybridArray,
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
        
        /// Load an instance of the templated image type
        /// from a file source, with specified reading options
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory>
        std::unique_ptr<ImageType> load(char const* source,
                                        options_map const& opts) {
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<im::FileSource> input;
            std::unique_ptr<im::Image> output;
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
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            if (!can_read) {
                PyErr_Format(PyExc_ValueError,
                    "Unimplemented read() in I/O format %s",
                    format->get_mimetype());
                return std::unique_ptr<ImageType>(nullptr);
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
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            {
                py::gil::release nogil;
                default_opts = format->add_options(opts);
                output = format->read(input.get(), &factory, default_opts);
                return im::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
        }
        
        /// Load an instance of the templated image type from a Py_buffer
        /// describing an in-memory source, with specified reading options
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory>
        std::unique_ptr<ImageType> loadblob(Py_buffer const& view,
                                            options_map const& opts) {
            FactoryType factory;
            std::unique_ptr<im::ImageFormat> format;
            std::unique_ptr<py::buffer::source> input;
            std::unique_ptr<im::Image> output;
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
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            if (!can_read) {
                PyErr_Format(PyExc_ValueError,
                    "Unimplemented read() in I/O format %s",
                    format->get_mimetype());
                return std::unique_ptr<ImageType>(nullptr);
            }
            
            {
                py::gil::release nogil;
                default_opts = format->add_options(opts);
                output = format->read(input.get(), &factory, default_opts);
                return im::detail::dynamic_cast_unique<ImageType>(
                    std::move(output));
            }
        }
        
        /// __init__ implementation
        template <typename ImageType = HybridArray,
                  typename FactoryType = ArrayFactory,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int init(PyObject* self, PyObject* args, PyObject* kwargs) {
            static const std::unique_ptr<ImageType> unique_null_ptr = std::unique_ptr<ImageType>(nullptr);
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            PyObject* py_is_blob = NULL;
            PyObject* options = NULL;
            Py_buffer view;
            char const* keywords[] = { "source", "is_blob", "options", NULL };
            bool is_blob = false;
            
            if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "s*|OO", const_cast<char**>(keywords),
                &view,                      /// "source", buffer with file path or image data
                &py_is_blob,                /// "is_blob", Python boolean specifying blobbiness
                &options))                  /// "options", read-options dict
            {
                PyErr_SetString(PyExc_ValueError,
                    "Bad arguments to image_init");
                return -1;
            } else {
                if (py_is_blob) {
                    /// test is necessary, the next line chokes on NULL:
                    is_blob = PyObject_IsTrue(py_is_blob);
                }
            }
            
            try {
                options_map opts = py::options::parse_options(options);
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
                /// there was something weird in the `options` dict
                PyErr_SetString(PyExc_AttributeError, exc.what());
                return -1;
            } catch (im::NotImplementedError& exc) {
                /// this shouldn't happen -- a generic ImageFormat pointer
                /// was returned when determining the blob image type
                PyErr_SetString(PyExc_AttributeError, exc.what());
                return -1;
            }
            if (pyim->image == unique_null_ptr) {
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
            
            /// ALL IS WELL:
            return 0;
        }
        
        /// __repr__ implementation
        template <typename ImageType = HybridArray,
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
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
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
        template <typename ImageType = HybridArray,
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
        template <typename ImageType = HybridArray,
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
        
        template <typename ImageType = HybridArray,
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
        
        template <typename ImageType = HybridArray,
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
        
        template <typename ImageType = HybridArray,
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
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        void dealloc(PyObject* self) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            pyim->cleanup();
            self->ob_type->tp_free(self);
        }
        
        template <typename ImageType = HybridArray>
        bool save(ImageType& input, char const* destination,
                                    options_map const& opts) {
            std::unique_ptr<im::ImageFormat> format;
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
                PyErr_Format(PyExc_ValueError,
                    "Unimplemented write() in I/O format %s",
                    format->get_mimetype());
                return false;
            }
            
            {
                py::gil::release nogil;
                exists = path::exists(destination);
                overwrite = opts.cast<bool>("overwrite", true);
            }
            
            if (exists && !overwrite) {
                PyErr_Format(PyExc_ValueError,
                    "File exists (opts['overwrite'] == true): %s", destination);
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
        
        /// Save an instance of the templated image from a Py_buffer
        /// describing an in-memory source, with specified reading options
        // template <typename ImageType = HybridArray>
        // std::unique_ptr<ImageType> saveblob(ImageType& input,
        //                                     Py_buffer& view,
        //                                     options_map const& opts) {
        //     std::unique_ptr<im::ImageFormat> format;
        //     std::unique_ptr<py::buffer::sink> output;
        //     options_map default_opts;
        //     try {
        //         py::gil::release nogil;
        //         output = std::unique_ptr<py::buffer::sink>(
        //             new py::buffer::sink(view));
        //         format = std::unique_ptr<im::ImageFormat>(
        //             im::get_format(opts.cast<char const*>("format")));
        //         default_opts = format->add_options(opts);
        //         format->write(dynamic_cast<Image&>(input),
        //                       output.get(), default_opts);
        //         return true;
        //     } catch (im::FormatNotFound& exc) {
        //         PyErr_SetString(PyExc_ValueError, exc.what());
        //         return false;
        //     }
        // }
        
        template <typename ImageType = HybridArray,
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
                PyErr_SetString(PyExc_ValueError,
                    "Bad arguments to write");
                return NULL;
            } else {
                if (py_as_blob) {
                    /// test is necessary, the next line chokes on NULL:
                    as_blob = PyObject_IsTrue(py_as_blob);
                }
            }
            
            try {
                options_map opts = py::options::parse_options(options);
                if (pyim->writeoptDict) {
                    opts = opts.update(pyim->writeopts());
                }
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
                did_save = py::image::save(*pyim->image.get(), dststr.c_str(), opts);
            } catch (im::OptionsError& exc) {
                /// there was something weird in the `options` dict
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
            
            if (as_blob && dststr.size()) {
                std::vector<byte> data;
                bool removed = false;
                {
                    py::gil::release nogil;
                    std::unique_ptr<im::FileSource> readback(
                        new im::FileSource(dststr.c_str()));
                    data = readback->full_data();
                    readback->close();
                    readback.reset(nullptr);
                }
                PyObject* out = PyString_FromStringAndSize(
                    (char const*)&data[0],
                    data.size());
                {
                    py::gil::release nogil;
                    removed = path::remove(dststr);
                }
                if (!removed) {
                    PyErr_Format(PyExc_ValueError,
                        "Failed to remove temporary file %s", dststr.c_str());
                    return NULL;
                }
                if (out == NULL) {
                    PyErr_SetString(PyExc_ValueError,
                        "Failed converting output to Python string");
                }
                return out;
            } else if (dststr.size()) {
                return PyString_FromString(dststr.c_str());
            }
            return Py_BuildValue("");
        }
        
        /// NumpyImage.dtype getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_dtype(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->dtype);
        }
        
        /// NumpyImage.shape getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_shape(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image->dim(0));
                case 2:
                    return Py_BuildValue("(ii)",    image->dim(0),
                                                    image->dim(1));
                case 3:
                    return Py_BuildValue("(iii)",   image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2),
                                                    image->dim(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image->dim(0),
                                                    image->dim(1),
                                                    image->dim(2),
                                                    image->dim(3),
                                                    image->dim(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        /// NumpyImage.strides getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_strides(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            auto image = pyim->image.get();
            switch (image->ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image->stride(0));
                case 2:
                    return Py_BuildValue("(ii)",    image->stride(0),
                                                    image->stride(1));
                case 3:
                    return Py_BuildValue("(iii)",   image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2),
                                                    image->stride(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image->stride(0),
                                                    image->stride(1),
                                                    image->stride(2),
                                                    image->stride(3),
                                                    image->stride(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        /// NumpyImage.read_opts getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_read_opts(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->readoptDict);
        }
        
        /// NumpyImage.read_opts setter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int          set_read_opts(PyObject* self, PyObject* value, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            if (!value) { return 0; }
            if (!PyMapping_Check(value)) {
                PyErr_SetString(PyExc_AttributeError,
                    "read_opts must be dict-ish");
                return -1;
            }
            Py_CLEAR(pyim->readoptDict);
            pyim->readoptDict = Py_BuildValue("O", value);
            return 0;
        }
        
        /// NumpyImage.write_opts getter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        PyObject*    get_write_opts(PyObject* self, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            return Py_BuildValue("O", pyim->writeoptDict);
        }
        
        /// NumpyImage.write_opts setter
        template <typename ImageType = HybridArray,
                  typename PythonImageType = PythonImageBase<ImageType>>
        int          set_write_opts(PyObject* self, PyObject* value, void* closure) {
            PythonImageType* pyim = reinterpret_cast<PythonImageType*>(self);
            if (!value) { return 0; }
            if (!PyMapping_Check(value)) {
                PyErr_SetString(PyExc_AttributeError,
                    "write_opts must be dict-ish");
                return -1;
            }
            Py_CLEAR(pyim->writeoptDict);
            pyim->writeoptDict = Py_BuildValue("O", value);
            return 0;
        }
        
    } /* namespace image */
        
} /* namespace py */

/// PYTHON TYPE DEFINITION

using im::HybridArray;
using im::ArrayFactory;
using py::image::NumpyImage;

static PyBufferProcs NumpyImage_Buffer3000Methods = {
    0, /* (readbufferproc) */
    0, /* (writebufferproc) */
    0, /* (segcountproc) */
    0, /* (charbufferproc) */
    (getbufferproc)py::image::getbuffer<HybridArray>,
    (releasebufferproc)py::image::releasebuffer<HybridArray>,
};

static PySequenceMethods NumpyImage_SequenceMethods = {
    (lenfunc)py::image::length<HybridArray>,        /* sq_length */
    0,                                              /* sq_concat */
    0,                                              /* sq_repeat */
    (ssizeargfunc)py::image::atindex<HybridArray>,  /* sq_item */
    0,                                              /* sq_slice */
    0,                                              /* sq_ass_item HAHAHAHA */
    0,                                              /* sq_ass_slice HEHEHE ASS <snort> HA */
    0                                               /* sq_contains */
};

static PyGetSetDef NumpyImage_getset[] = {
    {
        (char*)"dtype",
            (getter)py::image::get_dtype<HybridArray>,
            NULL,
            (char*)"NumpyImage dtype", NULL },
    {
        (char*)"shape",
            (getter)py::image::get_shape<HybridArray>,
            NULL,
            (char*)"NumpyImage shape tuple", NULL },
    {
        (char*)"strides",
            (getter)py::image::get_strides<HybridArray>,
            NULL,
            (char*)"NumpyImage strides tuple", NULL },
    {
        (char*)"read_opts",
            (getter)py::image::get_read_opts<HybridArray>,
            (setter)py::image::set_read_opts<HybridArray>,
            (char*)"Read options dict", NULL },
    {
        (char*)"write_opts",
            (getter)py::image::get_write_opts<HybridArray>,
            (setter)py::image::set_write_opts<HybridArray>,
            (char*)"Write options dict", NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyMethodDef NumpyImage_methods[] = {
    {
        "write",
            (PyCFunction)py::image::write<HybridArray>,
            METH_VARARGS | METH_KEYWORDS,
            "Format and write image data to file or blob" },
    { NULL, NULL, 0, NULL }
};

static Py_ssize_t NumpyImage_TypeFlags = Py_TPFLAGS_DEFAULT         | 
                                         Py_TPFLAGS_BASETYPE        | 
                                         Py_TPFLAGS_HAVE_NEWBUFFER;

static PyTypeObject NumpyImage_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                                                  /* ob_size */
    "im.NumpyImage",                                                    /* tp_name */
    sizeof(NumpyImage),                                                 /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::image::dealloc<HybridArray>,                        /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    0,                                                                  /* tp_compare */
    (reprfunc)py::image::repr<HybridArray>,                             /* tp_repr */
    0,                                                                  /* tp_as_number */
    &NumpyImage_SequenceMethods,                                        /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::image::hash<HybridArray>,                             /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::image::str<HybridArray>,                              /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    &NumpyImage_Buffer3000Methods,                                      /* tp_as_buffer */
    NumpyImage_TypeFlags,                                               /* tp_flags */
    "Python bindings for NumPy Halide bridge",                          /* tp_doc */
    0,                                                                  /* tp_traverse */
    0,                                                                  /* tp_clear */
    0,                                                                  /* tp_richcompare */
    0,                                                                  /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    NumpyImage_methods,                                                 /* tp_methods */
    0,                                                                  /* tp_members */
    NumpyImage_getset,                                                  /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::image::init<HybridArray, ArrayFactory>,               /* tp_init */
    0,                                                                  /* tp_alloc */
    py::image::createnew<HybridArray>,                                  /* tp_new */
    0,                                                                  /* tp_free */
    0,                                                                  /* tp_is_gc */
    0,                                                                  /* tp_bases */
    0,                                                                  /* tp_mro */
    0,                                                                  /* tp_cache */
    0,                                                                  /* tp_subclasses */
    0,                                                                  /* tp_weaklist */
    0,                                                                  /* tp_del */
    0,                                                                  /* tp_version_tag */
    
};

namespace py {
    
    namespace functions {
        
        PyObject* structcode_parse(PyObject* self, PyObject* args);
        
    }
}

static PyMethodDef NumpyImage_module_functions[] = {
    {
        "structcode_parse",
            (PyCFunction)py::functions::structcode_parse,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples" },
    { NULL, NULL, 0, NULL }
};

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_NUMPYIMAGE_HH_
