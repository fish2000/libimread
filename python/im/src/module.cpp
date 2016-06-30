
#include <cstddef>
#include <iostream>
#include "module.hpp"

PyTypeObject HybridImageModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                  /* ob_size */
    "im.HybridImage",                                                   /* tp_name */
    sizeof(HybridImageModel),                                           /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::image::dealloc<HalideNumpyImage>,                   /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    0,                                                                  /* tp_compare */
    (reprfunc)py::image::repr<HalideNumpyImage>,                        /* tp_repr */
    0,                                                                  /* tp_as_number */
    &HybridImage_SequenceMethods,                                       /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::image::hash<HalideNumpyImage>,                        /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::image::str<HalideNumpyImage>,                         /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    &HybridImage_Buffer3000Methods,                                     /* tp_as_buffer */
    HybridImageModel_TypeFlags,                                         /* tp_flags */
    "Python bindings for NumPy Halide bridge",                          /* tp_doc */
    0,                                                                  /* tp_traverse */
    0,                                                                  /* tp_clear */
    0,                                                                  /* tp_richcompare */
    0,                                                                  /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    HybridImage_methods,                                                /* tp_methods */
    0,                                                                  /* tp_members */
    HybridImage_getset,                                                 /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::image::init<HalideNumpyImage, HybridFactory>,         /* tp_init */
    0,                                                                  /* tp_alloc */
    py::image::createnew<HalideNumpyImage>,                             /* tp_new */
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

using im::HalideNumpyImage;
using im::ArrayImage;
using im::HybridFactory;
using im::ArrayFactory;
using py::ext::ImageModel;
using py::ext::ArrayModel;
using py::ext::ImageBufferModel;
using py::ext::ArrayBufferModel;

PyTypeObject BufferModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                  /* ob_size */
    py::ext::BufferModelBase<buffer_t>::typestring(),                   /* tp_name */
    sizeof(py::ext::BufferModelBase<buffer_t>),                         /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::ext::buffer::dealloc<buffer_t>,                     /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    (cmpfunc)py::ext::buffer::compare<buffer_t>,                        /* tp_compare */
    (reprfunc)py::ext::buffer::repr<buffer_t>,                          /* tp_repr */
    0,                                                                  /* tp_as_number */
    &Buffer_SequenceMethods,                                            /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    0,                                                                  /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::ext::buffer::str<buffer_t>,                           /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    &Buffer_Buffer3000Methods,                                          /* tp_as_buffer */
    py::ext::BufferModelBase<buffer_t>::typeflags(),                    /* tp_flags */
    py::ext::BufferModelBase<buffer_t>::typedoc(),                      /* tp_doc */
    (traverseproc)py::ext::buffer::traverse<buffer_t>,                  /* tp_traverse */
    (inquiry)py::ext::buffer::clear<buffer_t>,                          /* tp_clear */
    0,                                                                  /* tp_richcompare */
    py::detail::offset(&py::ext::BufferModelBase<buffer_t>::weakrefs),  /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    Buffer_methods,                                                     /* tp_methods */
    0,                                                                  /* tp_members */
    Buffer_getset,                                                      /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::ext::buffer::init<buffer_t>,                          /* tp_init */
    0,                                                                  /* tp_alloc */
    py::ext::buffer::createnew<buffer_t>,                               /* tp_new */
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

PyTypeObject ImageModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                  /* ob_size */
    py::ext::ImageModel::typestring(),                                  /* tp_name */
    sizeof(ImageModel),                                                 /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::ext::image::dealloc<HalideNumpyImage, buffer_t>,    /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    (cmpfunc)py::ext::image::compare<HalideNumpyImage, buffer_t>,       /* tp_compare */
    (reprfunc)py::ext::image::repr<HalideNumpyImage, buffer_t>,         /* tp_repr */
    0,                                                                  /* tp_as_number */
    py::ext::image::methods::sequence<HalideNumpyImage>(),              /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::ext::image::hash<HalideNumpyImage, buffer_t>,         /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::ext::image::str<HalideNumpyImage, buffer_t>,          /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    py::ext::image::methods::buffer<HalideNumpyImage>(),                /* tp_as_buffer */
    py::ext::ImageModel::typeflags(),                                   /* tp_flags */
    py::ext::ImageModel::typedoc(),                                     /* tp_doc */
    (traverseproc)py::ext::image::traverse<HalideNumpyImage, buffer_t>, /* tp_traverse */
    (inquiry)py::ext::image::clear<HalideNumpyImage, buffer_t>,         /* tp_clear */
    0,                                                                  /* tp_richcompare */
    py::detail::offset(&ImageModel::weakrefs),                          /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    py::ext::image::methods::basic<HalideNumpyImage>(),                 /* tp_methods */
    0,                                                                  /* tp_members */
    py::ext::image::methods::getset<HalideNumpyImage>(),                /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::ext::image::init<HalideNumpyImage, buffer_t>,         /* tp_init */
    0,                                                                  /* tp_alloc */
    py::ext::image::createnew<HalideNumpyImage, buffer_t>,              /* tp_new */
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

PyTypeObject ImageBufferModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                      /* ob_size */
    py::ext::ImageModel::BufferModel::typestring(),                         /* tp_name */
    sizeof(ImageBufferModel),                                               /* tp_basicsize */
    0,                                                                      /* tp_itemsize */
    (destructor)py::ext::buffer::dealloc<buffer_t, ImageBufferModel>,       /* tp_dealloc */
    0,                                                                      /* tp_print */
    0,                                                                      /* tp_getattr */
    0,                                                                      /* tp_setattr */
    (cmpfunc)py::ext::buffer::compare<buffer_t, ImageBufferModel>,          /* tp_compare */
    (reprfunc)py::ext::buffer::repr<buffer_t, ImageBufferModel>,            /* tp_repr */
    0,                                                                      /* tp_as_number */
    py::ext::buffer::methods::sequence<buffer_t, ImageBufferModel>(),       /* tp_as_sequence */
    0,                                                                      /* tp_as_mapping */
    0,                                                                      /* tp_hash */
    0,                                                                      /* tp_call */
    (reprfunc)py::ext::buffer::str<buffer_t, ImageBufferModel>,             /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                                  /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                                  /* tp_setattro */
    py::ext::buffer::methods::buffer<buffer_t, ImageBufferModel>(),         /* tp_as_buffer */
    py::ext::ImageModel::BufferModel::typeflags(),                          /* tp_flags */
    py::ext::ImageModel::BufferModel::typedoc(),                            /* tp_doc */
    (traverseproc)py::ext::buffer::traverse<buffer_t, ImageBufferModel>,    /* tp_traverse */
    (inquiry)py::ext::buffer::clear<buffer_t, ImageBufferModel>,            /* tp_clear */
    0,                                                                      /* tp_richcompare */
    py::detail::offset(&ImageBufferModel::weakrefs),                        /* tp_weaklistoffset */
    0,                                                                      /* tp_iter */
    0,                                                                      /* tp_iternext */
    py::ext::buffer::methods::basic<buffer_t, ImageBufferModel>(),          /* tp_methods */
    0,                                                                      /* tp_members */
    py::ext::buffer::methods::getset<buffer_t, ImageBufferModel>(),         /* tp_getset */
    0,                                                                      /* tp_base */
    0,                                                                      /* tp_dict */
    0,                                                                      /* tp_descr_get */
    0,                                                                      /* tp_descr_set */
    0,                                                                      /* tp_dictoffset */
    (initproc)py::ext::buffer::init<buffer_t, ImageBufferModel>,            /* tp_init */
    0,                                                                      /* tp_alloc */
    py::ext::buffer::createnew<buffer_t, ImageBufferModel>,                 /* tp_new */
    0,                                                                      /* tp_free */
    0,                                                                      /* tp_is_gc */
    0,                                                                      /* tp_bases */
    0,                                                                      /* tp_mro */
    0,                                                                      /* tp_cache */
    0,                                                                      /* tp_subclasses */
    0,                                                                      /* tp_weaklist */
    0,                                                                      /* tp_del */
    0,                                                                      /* tp_version_tag */
};

PyTypeObject ArrayModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                  /* ob_size */
    py::ext::ArrayModel::typestring(),                                  /* tp_name */
    sizeof(ArrayModel),                                                 /* tp_basicsize */
    0,                                                                  /* tp_itemsize */
    (destructor)py::ext::image::dealloc<ArrayImage, buffer_t>,          /* tp_dealloc */
    0,                                                                  /* tp_print */
    0,                                                                  /* tp_getattr */
    0,                                                                  /* tp_setattr */
    (cmpfunc)py::ext::image::compare<ArrayImage, buffer_t>,             /* tp_compare */
    (reprfunc)py::ext::image::repr<ArrayImage, buffer_t>,               /* tp_repr */
    0,                                                                  /* tp_as_number */
    py::ext::image::methods::sequence<ArrayImage>(),                    /* tp_as_sequence */
    0,                                                                  /* tp_as_mapping */
    (hashfunc)py::ext::image::hash<ArrayImage, buffer_t>,               /* tp_hash */
    0,                                                                  /* tp_call */
    (reprfunc)py::ext::image::str<ArrayImage, buffer_t>,                /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                              /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                              /* tp_setattro */
    py::ext::image::methods::buffer<ArrayImage>(),                      /* tp_as_buffer */
    py::ext::ArrayModel::typeflags(),                                   /* tp_flags */
    py::ext::ArrayModel::typedoc(),                                     /* tp_doc */
    (traverseproc)py::ext::image::traverse<ArrayImage, buffer_t>,       /* tp_traverse */
    (inquiry)py::ext::image::clear<ArrayImage, buffer_t>,               /* tp_clear */
    0,                                                                  /* tp_richcompare */
    py::detail::offset(&ArrayModel::weakrefs),                          /* tp_weaklistoffset */
    0,                                                                  /* tp_iter */
    0,                                                                  /* tp_iternext */
    py::ext::image::methods::basic<ArrayImage>(),                       /* tp_methods */
    0,                                                                  /* tp_members */
    py::ext::image::methods::getset<ArrayImage>(),                      /* tp_getset */
    0,                                                                  /* tp_base */
    0,                                                                  /* tp_dict */
    0,                                                                  /* tp_descr_get */
    0,                                                                  /* tp_descr_set */
    0,                                                                  /* tp_dictoffset */
    (initproc)py::ext::image::init<ArrayImage, buffer_t>,               /* tp_init */
    0,                                                                  /* tp_alloc */
    py::ext::image::createnew<ArrayImage, buffer_t>,                    /* tp_new */
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

PyTypeObject ArrayBufferModel_Type = {
    PyObject_HEAD_INIT(nullptr)
    0,                                                                      /* ob_size */
    py::ext::ArrayModel::BufferModel::typestring(),                         /* tp_name */
    sizeof(ArrayBufferModel),                                               /* tp_basicsize */
    0,                                                                      /* tp_itemsize */
    (destructor)py::ext::buffer::dealloc<buffer_t, ArrayBufferModel>,       /* tp_dealloc */
    0,                                                                      /* tp_print */
    0,                                                                      /* tp_getattr */
    0,                                                                      /* tp_setattr */
    (cmpfunc)py::ext::buffer::compare<buffer_t, ArrayBufferModel>,          /* tp_compare */
    (reprfunc)py::ext::buffer::repr<buffer_t, ArrayBufferModel>,            /* tp_repr */
    0,                                                                      /* tp_as_number */
    py::ext::buffer::methods::sequence<buffer_t, ArrayBufferModel>(),       /* tp_as_sequence */
    0,                                                                      /* tp_as_mapping */
    0,                                                                      /* tp_hash */
    0,                                                                      /* tp_call */
    (reprfunc)py::ext::buffer::str<buffer_t, ArrayBufferModel>,             /* tp_str */
    (getattrofunc)PyObject_GenericGetAttr,                                  /* tp_getattro */
    (setattrofunc)PyObject_GenericSetAttr,                                  /* tp_setattro */
    py::ext::buffer::methods::buffer<buffer_t, ArrayBufferModel>(),         /* tp_as_buffer */
    py::ext::ArrayModel::BufferModel::typeflags(),                          /* tp_flags */
    py::ext::ArrayModel::BufferModel::typedoc(),                            /* tp_doc */
    (traverseproc)py::ext::buffer::traverse<buffer_t, ArrayBufferModel>,    /* tp_traverse */
    (inquiry)py::ext::buffer::clear<buffer_t, ArrayBufferModel>,            /* tp_clear */
    0,                                                                      /* tp_richcompare */
    py::detail::offset(&ArrayBufferModel::weakrefs),                        /* tp_weaklistoffset */
    0,                                                                      /* tp_iter */
    0,                                                                      /* tp_iternext */
    py::ext::buffer::methods::basic<buffer_t, ArrayBufferModel>(),          /* tp_methods */
    0,                                                                      /* tp_members */
    py::ext::buffer::methods::getset<buffer_t, ArrayBufferModel>(),         /* tp_getset */
    0,                                                                      /* tp_base */
    0,                                                                      /* tp_dict */
    0,                                                                      /* tp_descr_get */
    0,                                                                      /* tp_descr_set */
    0,                                                                      /* tp_dictoffset */
    (initproc)py::ext::buffer::init<buffer_t, ArrayBufferModel>,            /* tp_init */
    0,                                                                      /* tp_alloc */
    py::ext::buffer::createnew<buffer_t, ArrayBufferModel>,                 /* tp_new */
    0,                                                                      /* tp_free */
    0,                                                                      /* tp_is_gc */
    0,                                                                      /* tp_bases */
    0,                                                                      /* tp_mro */
    0,                                                                      /* tp_cache */
    0,                                                                      /* tp_subclasses */
    0,                                                                      /* tp_weaklist */
    0,                                                                      /* tp_del */
    0,                                                                      /* tp_version_tag */
};

static PyMethodDef module_functions[] = {
    {
        "detect",
            (PyCFunction)py::functions::detect,
            METH_VARARGS | METH_KEYWORDS,
            "detect(source=\"\", file=None, is_blob=False)\n"
            "\t-> Detect the format of a file, blob, or file-like Python object\n"
            "\t   specifying: \n"
            "\t - a path or blob (source) OR \n"
            "\t - a filehandle opened for reading (file) AND OPTIONALLY \n"
            "\t - a boolean flag indicating data to be read as bytes is passed in (is_blob)\n" },
    {
        "structcode_parse",
            (PyCFunction)py::functions::structcode_parse,
            METH_VARARGS,
            "Parse struct code into list of dtype-string tuples" },
    {
        "structcode_convert",
            (PyCFunction)py::functions::structcode_convert,
            METH_VARARGS,
            "Directly convert struct code internal tuple to Python values" },
    {
        "hybridimage_check",
            (PyCFunction)py::functions::hybridimage_check,
            METH_VARARGS,
            "Boolean function to test for HybridImage instances" },
    {
        "image_check",
            (PyCFunction)py::functions::image_check,
            METH_VARARGS,
            "Boolean function to test for im.Image instances" },
    {
        "buffer_check",
            (PyCFunction)py::functions::buffer_check,
            METH_VARARGS,
            "Boolean function to test for im.Buffer instances" },
    {
        "imagebuffer_check",
            (PyCFunction)py::functions::imagebuffer_check,
            METH_VARARGS,
            "Boolean function to test for im.Image.Buffer instances" },
    {
        "array_check",
            (PyCFunction)py::functions::array_check,
            METH_VARARGS,
            "Boolean function to test for im.Array instances" },
    {
        "arraybuffer_check",
            (PyCFunction)py::functions::arraybuffer_check,
            METH_VARARGS,
            "Boolean function to test for im.Array.Buffer instances" },
    { nullptr, nullptr, 0, nullptr }
};

PyDoc_STRVAR(module__doc__,     "Python bindings for libimread");
PyDoc_STRVAR(module__name__,    "im.im");

namespace {
    
    bool initialize(PyObject* module) {
        /// Bring in filesystem::path
        using filesystem::path;
        
        /// Declare some object pointers:
        /// ... one to the tuple of image-format suffix strings …
        PyObject* format_tuple;
        
        /// ... another to the root dict of image-format info dicts …
        PyObject* format_infodict;
        
        /// ... two more for endianness …
        PyObject* _byteorder;
        PyObject* _byteordermark;
        
        /// Nix the obsequeously scrupulous global I/O buffer behavior
        /// ... doctor's orders, q.v. http://ubm.io/1VCwhdy
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(0);
        
        /// Initialize Python threads and GIL state
        PyEval_InitThreads();
        
        /// Bring in NumPy's C-API
        if (_import_array() < 0)                      { return false; }
        
        /// Manually amend our declared types, as needed:
        /// -- Specify that im.ImageBuffer subclasses im.Buffer
        ImageBufferModel_Type.tp_base = &BufferModel_Type;
        ArrayBufferModel_Type.tp_base = &BufferModel_Type;
        
        /// -- Prepare im.Image.__dict__ for our customizations
        ImageModel_Type.tp_dict = PyDict_New();
        if (!ImageModel_Type.tp_dict)                 { return false; }
        ArrayModel_Type.tp_dict = PyDict_New();
        if (!ArrayModel_Type.tp_dict)                 { return false; }
        
        /// Check readiness of our extension type declarations (?)
        if (PyType_Ready(&HybridImageModel_Type) < 0) { return false; }
        if (PyType_Ready(&BufferModel_Type) < 0)      { return false; }
        if (PyType_Ready(&ImageModel_Type) < 0)       { return false; }
        if (PyType_Ready(&ImageBufferModel_Type) < 0) { return false; }
        if (PyType_Ready(&ArrayModel_Type) < 0)       { return false; }
        if (PyType_Ready(&ArrayBufferModel_Type) < 0) { return false; }
        
        /// Get the path of the extension module --
        /// via dladdr() on the module init function's address
        path modulefile((const void*)initialize);
        
        /// Set the module object's __file__ attribute
        PyObject_SetAttrString(module,
            "__file__",
            py::string(modulefile.make_absolute().str()));
        
        /// Set the module object's __path__ single-element list
        PyObject_SetAttrString(module,
            "__path__",
            py::listify(modulefile.make_absolute().parent().str()));
        
        /// Add the HybridImageModel type object to the module
        Py_INCREF(&HybridImageModel_Type);
        PyModule_AddObject(module,
            "HybridImage",
            (PyObject*)&HybridImageModel_Type);
        
        /// Add the BufferModel type object to the module
        Py_INCREF(&BufferModel_Type);
        PyModule_AddObject(module,
            "Buffer",
            (PyObject*)&BufferModel_Type);
        
        /// Add the ImageBufferModel type object directly to im.Image.__dict__,
        /// such that ImageBuffer presents as an inner class of im.Image, e.g.
        /// 
        ///     im.Image.ImageBuffer
        /// 
        /// ... thanks SO! http://stackoverflow.com/q/35954016/298171
        Py_INCREF(&ImageBufferModel_Type);
        PyDict_SetItemString(ImageModel_Type.tp_dict,
            "Buffer",
            (PyObject*)&ImageBufferModel_Type);
        
        /// Add the ImageModel type object to the module
        Py_INCREF(&ImageModel_Type);
        PyModule_AddObject(module,
            "Image",
            (PyObject*)&ImageModel_Type);
        
        Py_INCREF(&ArrayBufferModel_Type);
        PyDict_SetItemString(ArrayModel_Type.tp_dict,
            "Buffer",
            (PyObject*)&ArrayBufferModel_Type);
        
        /// Add the ImageModel type object to the module
        Py_INCREF(&ArrayModel_Type);
        PyModule_AddObject(module,
            "Array",
            (PyObject*)&ArrayModel_Type);
        
        /// Get the master list of image format suffixes,
        /// newly formatted as a Python tuple of strings,
        /// and add this to the module as a static-ish constant
        format_tuple = py::detail::formats_as_pytuple();
        if (format_tuple == nullptr)                  { return false; }
        PyModule_AddObject(module,
            "formats",
            format_tuple);
        
        format_infodict = py::detail::formats_as_infodict();
        if (format_infodict == nullptr)               { return false; }
        PyModule_AddObject(module,
            "format_info",
            PyDictProxy_New(format_infodict));
        
        /// Store the byte order of the system in im._byteorder and im._byteordermark
        /// ... note that the byte-order-determining function (in hybrid.cpp) uses
        /// the exact same logic used in the python `sys` module's implementation
        _byteorder = py::string(im::byteorder == im::Endian::Big ? "big" : "little");
        _byteordermark = py::string((char)im::byteorder);
        if (_byteorder == nullptr)                    { return false; }
        if (_byteordermark == nullptr)                { return false; }
        PyModule_AddObject(module,
            "_byteorder",
            _byteorder);
        PyModule_AddObject(module,
            "_byteordermark",
            _byteordermark);
        
        /// all is well:
        return true;
    }
    
}

#if PY_VERSION_HEX >= 0x03000000
#pragma mark - Python 3+ Module Initializer
PyMODINIT_FUNC PyInit_im(void) {
    PyObject* module;
    
    /// Static module-definition table
    static PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        module__name__,                           /* m_name */
        module__doc__,                            /* m_doc */
        -1,                                       /* m_size */
        module_functions                          /* m_methods */
    };
    
    /// Actually initialize the module object,
    /// using the new Python 3 module-definition table
    module = PyModule_Create(&moduledef);
    
    /// Initialize and check module
    if (module == nullptr)                        { return nullptr; }
    if (!initialize(module))                      { return nullptr; }
    
    /// Return module object
    return module;
}
#else
#pragma mark - Python 2 Module Initializer
PyMODINIT_FUNC initim(void) {
    PyObject* module;
    
    /// Actually initialize the module object,
    /// setting up a module C-function table
    module = Py_InitModule3(
        module__name__,
        module_functions,
        module__doc__);
    
    /// Initialize and check module
    if (module == nullptr)                        { return; }
    if (!initialize(module))                      { return; }
}
#endif