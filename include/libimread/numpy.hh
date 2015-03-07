// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_NUMPY_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <memory>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

#if PY_MAJOR_VERSION < 3
#define PyBytes_FromString(string) PyString_FromString(string)
#endif

namespace im {

    class NumpyImage : public Image, public ImageWithMetadata {
        public:
            NumpyImage(PyArrayObject *a = 0)
                :array(a)
                { }
            
            ~NumpyImage() { Py_XDECREF(array); }
            
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
                    return std::unique_ptr<Image>(new NumpyImage(array));
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
