
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include "hybrid.hh"
#include "gil.hpp"
#include "typecode.hpp"

namespace py {
    
    PyObject* None();
    PyObject* True();
    PyObject* False();
    
    PyObject* boolean(bool truth = false);
    PyObject* string(std::string const&);
    PyObject* string(char const*);
    PyObject* string(char);
    PyObject* object(PyObject* arg = nullptr);
    PyObject* object(PyArray_Descr* arg = nullptr);
    
    template <typename ...Args> inline
    PyObject* tuple(Args&& ...args) {
        static_assert(
            sizeof...(Args) > 0,
            "Can't pack zero-length argument list to PyTuple");
        
        return PyTuple_Pack(
            sizeof...(Args),
            std::forward<Args>(args)...);
    }
    
    namespace detail {
        
        /// Use nop<MyThing> to make a non-deleting unique_ptr e.g.
        /// 
        ///     using nondeleting_ptr = std::unique_ptr<MyThing,
        ///                             py::detail::nop<MyThing>>;
        /// 
        template <typename B>
        struct nop {
            constexpr nop() noexcept = default;
            template <typename U> nop(nop<U> const&) noexcept {}
            void operator()(std::add_pointer_t<B> ptr) { /*NOP*/ }
        };
        
        
        /// C++11 constexpr-friendly reimplementation of `offsetof()` --
        /// see also: https://gist.github.com/graphitemaster/494f21190bb2c63c5516
        /// 
        ///     std::size_t o = py::detail::offset(&ContainingType::classMember); /// or
        ///     std::size_t o = PYDEETS_OFFSET(ContainingType, classMember);
        /// 
        namespace {
            
            template <typename T1, typename T2>
            struct offset_impl {
                static T2 thing;
                static constexpr std::size_t off_by(T1 T2::*member) {
                    return std::size_t(&(offset_impl<T1, T2>::thing.*member)) -
                           std::size_t(&offset_impl<T1, T2>::thing);
                }
            };
            
            template <typename T1, typename T2>
            T2 offset_impl<T1, T2>::thing;
            
        }
        
        template <typename T1, typename T2> inline
        constexpr Py_ssize_t offset(T1 T2::*member) {
            return static_cast<Py_ssize_t>(
                offset_impl<T1, T2>::off_by(member));
        }
        
        #define PYDEETS_OFFSET(type, member) py::detail::offset(&type::member)
        
        /// XXX: remind me why in fuck did I write this shit originally
        template <typename T, typename pT>
        std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT>&& source) {
            /// Force a dynamic_cast upon a unique_ptr via interim swap
            /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
            /// ... from http://stackoverflow.com/a/14777419/298171
            if (!source) { return std::unique_ptr<T>(); }
            
            /// Throws a std::bad_cast() if this doesn't work out
            T *destination = &dynamic_cast<T&>(*source.get());
            
            source.release();
            std::unique_ptr<T> out(destination);
            return out;
        }
        
        /// shortcuts for getting tuples from images
        template <typename ImageType> inline
        PyObject* image_shape(ImageType const& image) {
            switch (image.ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image.dim(0));
                case 2:
                    return Py_BuildValue("(ii)",    image.dim(0),
                                                    image.dim(1));
                case 3:
                    return Py_BuildValue("(iii)",   image.dim(0),
                                                    image.dim(1),
                                                    image.dim(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image.dim(0),
                                                    image.dim(1),
                                                    image.dim(2),
                                                    image.dim(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image.dim(0),
                                                    image.dim(1),
                                                    image.dim(2),
                                                    image.dim(3),
                                                    image.dim(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        template <typename ImageType> inline
        PyObject* image_strides(ImageType const& image) {
            switch (image.ndims()) {
                case 1:
                    return Py_BuildValue("(i)",     image.stride(0));
                case 2:
                    return Py_BuildValue("(ii)",    image.stride(0),
                                                    image.stride(1));
                case 3:
                    return Py_BuildValue("(iii)",   image.stride(0),
                                                    image.stride(1),
                                                    image.stride(2));
                case 4:
                    return Py_BuildValue("(iiii)",  image.stride(0),
                                                    image.stride(1),
                                                    image.stride(2),
                                                    image.stride(3));
                case 5:
                    return Py_BuildValue("(iiiii)", image.stride(0),
                                                    image.stride(1),
                                                    image.stride(2),
                                                    image.stride(3),
                                                    image.stride(4));
                default:
                    return Py_BuildValue("");
            }
            return Py_BuildValue("");
        }
        
        /// A misnomer -- actually returns a dtype-compatible tuple full
        /// of label/format subtuples germane to the description
        /// of the parsed typecode you pass it
        PyObject* structcode_to_dtype(char const* code);
        
        // tuple((),()...)  py::detail::structcode_to_dtype(char)
        //   "NPY_UINT32"   typecode::name(NPY_TYPES)
        //            "b"   typecode::typechar(NPY_TYPES)
        //            "b"   im::detail::character_for<PixelType>()
        //            "B"   im::detail::structcode(NPY_TYPES)
        //        HalType   im::detail::for_dtype(NPY_TYPES)
        //        HalType   im::detail::for_type<uint32_t>()
        //          "|b8"   im::detail::encoding_for<PixelType>(e)
        //      NPY_TYPES   im::detail::for_nbits(nbits, is_signed=false)
        // py::detail::structcode_to_dtype(typecode) || nullptr
        
        template <typename ImageType> inline
        PyArrayInterface* array_struct(ImageType const& image,
                                       bool include_descriptor = true) {
            PyArrayInterface* out   = nullptr;
            void* data              = image.rowp(0);
            int ndims               = image.ndims();
            int flags               = include_descriptor ? NPY_ARR_HAS_DESCR : 0;
            NPY_TYPES typecode      = image.dtype();
            NPY_TYPECHAR typechar   = typecode::typechar(typecode);
            PyObject* descriptor    = nullptr;
            
            if (include_descriptor) {
                py::gil::ensure yesgil;
                descriptor = py::detail::structcode_to_dtype(
                             im::detail::structcode(typecode));
            }
            
            out = new PyArrayInterface {
                2,                          /// brought to you by
                ndims,
                (char)typechar,
                sizeof(uint8_t),            /// need to not hardcode this
                flags,
                new Py_intptr_t[ndims],     /// shape
                new Py_intptr_t[ndims],     /// strides
                data,                       /// void* data
                descriptor                  /// PyObject*
            };
            
            for (int idx = 0; idx < ndims; idx++) {
                out->shape[idx]   = image.dim(idx);
                out->strides[idx] = image.stride(idx);
            }
            
            return out;
        }
        
        /// One-and-done functions for dumping a tuple of python strings
        /// filled with valid (as in registered) format suffixes; currently
        /// we just call this the once in the extensions' init func and stash
        /// the return value in the `im` module PyObject, just in there like
        /// alongside the types and other junk
        using stringvec_t = std::vector<std::string>;
        stringvec_t& formats_as_vector();                   /// this one is GIL-optional (how european!)
        PyObject* formats_as_pytuple(int idx = 0);          /// whereas here, no GIL no shoes no funcall
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_