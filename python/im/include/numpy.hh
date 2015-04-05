// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <cstring>
#include <vector>
#include <memory>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

#if PY_MAJOR_VERSION < 3
#define PyBytes_FromString(string) PyString_FromString(string)
#endif

namespace im {
    
    namespace dtype {
        
        
        using typecode = NPY_TYPES;
        using Halide::Type;
        
        typecode from_typestruct(Type ts) {
            switch (ts.code) {
                
                case Type::UInt:
                    switch (ts.bits) {
                        case 1: return NPY_BOOL;
                        case 8: return NPY_UINT8;
                        case 16: return NPY_UINT16;
                        case 32: return NPY_UINT32;
                        case 64: return NPY_UINT64;
                    }
                    return NPY_NOTYPE;
            
                case Type::Int:
                switch (ts.bits) {
                    case 1: return NPY_BOOL;
                    case 8: return NPY_INT8;
                    case 16: return NPY_INT16;
                    case 32: return NPY_INT32;
                    case 64: return NPY_INT64;
                }
                return NPY_NOTYPE;
            
                case Type::Float:
                switch (ts.bits) {
                    // case 1: return NPY_BOOL;
                    // case 8: return NPY_UINT8;
                    case 16: return NPY_HALF;
                    case 32: return NPY_FLOAT32;
                    case 64: return NPY_FLOAT64;
                    // case WTF: return NPY_LONGDOUBLE ????
                }
                return NPY_NOTYPE;
        
                /// WELL FUCK
                case Type::Handle: return NPY_VOID;
            }
            return NPY_NOTYPE;
        }
        
        Type from_typecode(typecode tc) {
            Type t;
            if (PyArray_ValidType(tc) != NPY_TRUE) { return t; }
            if (PyTypeNum_ISBOOL(tc)) { return Halide::Bool(); }
            if (!PyTypeNum_ISNUMBER(tc)) { t.code = Type::Handle; }
            if (PyTypeNum_ISFLOAT(tc)) { t.code = Type::Float; }
            if (PyTypeNum_ISINTEGER(tc) && PyTypeNum_ISUNSIGNED(tc)) { t.code = Type::UInt; }
            if (PyTypeNum_ISINTEGER(tc) && PyTypeNum_ISSIGNED(tc)) { t.code = Type::UInt; }
            
        }
        
        
        switch (PyArray_TYPE(array)) {
            case NPY_UINT8:
            case NPY_INT8:
                return 8;
            case NPY_UINT16:
            case NPY_INT16:
                return 16;
            case NPY_UINT32:
            case NPY_INT32:
                return 32;
            case NPY_UINT64:
            case NPY_INT64:
                return 64;
            default:
                throw ProgrammingError();
        }
        
        PyTypeNum_ISUNSIGNED(number);
        PyDataType_ISUNSIGNED(dtypestruct);
        PyArray_ISUNSIGNED(array);
        
        
        PyTypeNum_ISINTEGER();
        PyTypeNum_ISFLOAT();
        PyTypeNum_ISBOOL();
        PyTypeNum_ISNUMBER();
        
        PyArray_ValidType() == NPY_TRUE
        
    }
    
    /// We use Halide::ImageBase instead of Halide::Image here,
    /// so that we don't have to muck around with templates when
    /// working with arbitrary NumPy dtype values.
    using HalBase = Halide::ImageBase;
    using MetaImage = ImageWithMetadata;
    
    class HybridArray : public HalBase, public Image, public MetaImage {
        public:
            HybridArray(PyArrayObject *a = 0)
                :HalBase(), Image(), MetaImage()
                ,array(a)
                { }
            
            using HalBase::dimensions;
            using HalBase::extent;
            using HalBase::stride;
            using HalBase::channels;
            using HalBase::raw_buffer;
            using HalBase::buffer;
            
            ~HybridArray() { Py_XDECREF(array); }
            
            PyArrayObject *release() {
                PyArrayObject* r = array;
                array = 0;
                return r;
            }
            
            PyObject *releasePyObject() {
                this->finalize();
                return reinterpret_cast<PyObject*>(this->release());
            }
            
            PyObject *metadataPyObject() {
                std::string* s = this->get_meta();
                if (s) { return PyBytes_FromString(s->c_str()); }
                Py_RETURN_NONE;
            }
            
            /// This returns the same type of data as buffer_t.host
            virtual uint8_t data(int s) const {
                return HalBase::buffer.host_ptr();
            }
            
            virtual int stride(int s) const override {
                return HalBase::stride(s);
            }
            
            virtual int nbits() const {
                if (!array) { throw ProgrammingError(); }
                switch (PyArray_TYPE(array)) {
                    case NPY_UINT8:
                    case NPY_INT8:
                        return 8;
                    case NPY_UINT16:
                    case NPY_INT16:
                        return 16;
                    case NPY_UINT32:
                    case NPY_INT32:
                        return 32;
                    case NPY_UINT64:
                    case NPY_INT64:
                        return 64;
                    default:
                        throw ProgrammingError();
                }
            }
            
            virtual int ndims() const {
                if (!array) { throw ProgrammingError(); }
                return PyArray_NDIM(array);
            }
            
            virtual int dim(int d) const {
                if (!array || d >= this->ndims()) { throw ProgrammingError(); }
                return PyArray_DIM(array, d);
            }
            
            virtual void *rowp(int r) {
                if (!array) throw ProgrammingError();
                if (r >= PyArray_DIM(array, 0)) { throw ProgrammingError(); }
                return PyArray_GETPTR1(array, r);
            }
            
            void finalize();
            PyArrayObject *array;
    };
    
    class NumpyFactory : public ImageFactory {
        protected:
            std::unique_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) {
                npy_intp dims[5];
                dims[0] = d0;
                dims[1] = d1;
                dims[2] = d2;
                dims[3] = d3;
                dims[4] = d4;
                npy_intp nd = 5;
                
                if (d2 == -1) nd = 2;
                else if (d3 == -1) nd = 3;
                else if (d4 == -1) nd = 4;
                int dtype = -1;
                switch (nbits) {
                    case 1: dtype = NPY_BOOL; break;
                    case 8: dtype = NPY_UINT8; break;
                    case 16: dtype = NPY_UINT16; break;
                    case 32: dtype = NPY_UINT32; break;
                    default: {
                        std::ostringstream out;
                        out << "im::NumpyFactory::create(): Cannot handle " << nbits << "-bit images.";
                        throw ProgrammingError(out.str());
                    }
                }
                
                PyArrayObject *array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims, dtype));
                if (!array) { throw std::bad_alloc(); }
                try {
                    return std::unique_ptr<Image>(new HybridArray(array));
                } catch (const std::exception &ex) {
                    Py_DECREF(array);
                    throw ex;
                } catch (...) {
                    Py_DECREF(array);
                    throw;
                }
            }
    };

}

#endif // LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
