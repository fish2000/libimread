
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_

#include <tuple>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <initializer_list>
#include <Python.h>
#include <libimread/libimread.hpp>

/// forward-declare PyArray_Descr from numpy
struct _PyArray_Descr;
typedef _PyArray_Descr PyArray_Descr;

namespace py {
    
    PyObject* None();
    PyObject* True();
    PyObject* False();
    
    PyObject* boolean(bool truth = false);
    PyObject* string(std::string const&);
    PyObject* string(char const*);
    PyObject* string(char const*, std::size_t);
    PyObject* string(char);
    PyObject* object(PyObject* arg = nullptr);
    PyObject* object(PyArray_Descr* arg = nullptr);
    
    template <typename ...Args> inline
    PyObject* tuple(Args&& ...args) {
        static_assert(
            sizeof...(Args) > 0,
            "Can't pack a zero-length arglist as a PyTuple");
        
        return PyTuple_Pack(
            sizeof...(Args),
            std::forward<Args>(args)...);
    }
    
    PyObject* convert(PyObject*);
    PyObject* convert(std::nullptr_t);
    PyObject* convert(bool);
    PyObject* convert(void*);
    PyObject* convert(std::size_t);
    PyObject* convert(Py_ssize_t);
    PyObject* convert(int8_t);
    PyObject* convert(int16_t);
    PyObject* convert(int32_t);
    PyObject* convert(int64_t);
    PyObject* convert(uint8_t);
    PyObject* convert(uint16_t);
    PyObject* convert(uint32_t);
    PyObject* convert(uint64_t);
    PyObject* convert(float);
    PyObject* convert(double);
    PyObject* convert(long double);
    PyObject* convert(char*);
    PyObject* convert(char const*);
    PyObject* convert(std::string const&);
    PyObject* convert(char*, std::size_t);
    PyObject* convert(char const*, std::size_t);
    
    template <typename Cast,
              typename Original,
              typename std::enable_if_t<std::is_arithmetic<Cast>::value &&
                                        std::is_arithmetic<Original>::value,
              int> = 0> inline
    PyObject* convert(Original orig) {
        return py::convert(static_cast<Cast>(orig));
    }
    
    template <typename Cast,
              typename Original,
              typename std::enable_if_t<!std::is_arithmetic<Cast>::value &&
                                        !std::is_arithmetic<Original>::value,
              int> = 0> inline
    PyObject* convert(Original orig) {
        return py::convert(reinterpret_cast<Cast>(orig));
    }
    
    template <typename Mapping,
              typename Value = typename Mapping::mapped_type,
              typename std::enable_if_t<
                       std::is_constructible<std::string,
                                             typename Mapping::key_type>::value,
              int> = 0> inline
    PyObject* convert(Mapping const& mapping) {
        PyObject* dict = PyDict_New();
        for (auto const& item : mapping) {
            std::string key(item.first);
            PyDict_SetItemString(dict, key.c_str(), py::convert(item.second));
        }
        return dict;
    }
    
    template <typename Vector,
              typename Value = typename Vector::value_type> inline
    PyObject* convert(Vector const& vector) {
        Py_ssize_t idx = 0,
                   max = vector.size();
        PyObject* tuple = PyTuple_New(max);
        for (Value const& item : vector) {
            PyTuple_SET_ITEM(tuple, idx, py::convert(item));
            idx++;
        }
        return tuple;
    }
    
    template <typename ...Args>
    using convertible = std::is_same<std::add_pointer_t<PyObject>,
                                     decltype(py::convert(Args{}...))>;
    
    template <typename ...Args>
    bool convertible_v = py::convertible<Args...>::value;
    
    PyObject* tuplize();
    
    template <typename ListType,
              typename std::enable_if_t<
                        py::convertible<ListType>::value,
              int> = 0>
    PyObject* tuplize(std::initializer_list<ListType> list) {
        Py_ssize_t idx = 0,
                   max = list.size();
        PyObject* tuple = PyTuple_New(max);
        for (ListType const& item : list) {
            PyTuple_SET_ITEM(tuple, idx, py::convert(item));
            idx++;
        }
        return tuple;
    }
    
    template <typename Args,
              typename std::enable_if_t<
                        py::convertible<Args>::value,
              int> = 0>
    PyObject* tuplize(Args arg) {
        return PyTuple_Pack(1,
            py::convert(std::forward<Args>(arg)));
    }
    
    template <typename ...Args>
    PyObject* tuplize(Args&& ...args) {
        static_assert(
            sizeof...(Args) > 1,
            "Can't tuplize a zero-length arglist");
        
        return PyTuple_Pack(
            sizeof...(Args),
            py::convert(std::forward<Args>(args))...);
    }
    
    PyObject* listify();
    
    template <typename ListType,
              typename std::enable_if_t<
                        py::convertible<ListType>::value,
              int> = 0>
    PyObject* listify(std::initializer_list<ListType> list) {
        Py_ssize_t idx = 0,
                   max = list.size();
        PyObject* pylist = PyList_New(max);
        for (ListType const& item : list) {
            PyList_SET_ITEM(pylist, idx, py::convert(item));
            idx++;
        }
        return pylist;
    }
    
    template <typename Args,
              typename std::enable_if_t<
                        py::convertible<Args>::value,
              int> = 0>
    PyObject* listify(Args arg) {
        PyObject* pylist = PyList_New(1);
        PyList_SET_ITEM(pylist, 0,
            py::convert(std::forward<Args>(arg)));
        return pylist;
    }
    
    template <typename Tuple, std::size_t ...I>
    PyObject* listify(Tuple&& t, std::index_sequence<I...>) {
        PyObject* pylist = PyList_New(
            std::tuple_size<Tuple>::value);
        unpack {
            PyList_SET_ITEM(pylist, I,
                py::convert(std::get<I>(std::forward<Tuple>(t))))...
        };
        return pylist;
    }
    
    template <typename ...Args>
    PyObject* listify(Args&& ...args) {
        using Indices = std::make_index_sequence<sizeof...(Args)>;
        static_assert(
            sizeof...(Args) > 1,
            "Can't tuplize a zero-length arglist");
        
        return py::listify(
            std::forward_as_tuple(args...),
            Indices());
    }
    
    namespace impl {
        
        template <typename F, typename Tuple, std::size_t ...I> inline
        auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
            return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
        }
        
        template <typename F, typename Tuple> inline
        auto apply(F&& f, Tuple&& t) {
            using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
            return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices());
        }
        
    }
    
    template <typename ...Args> inline
    PyObject* convert(std::tuple<Args&&...> argtuple) {
        using Tuple = std::tuple<Args&&...>;
        static_assert(
            std::tuple_size<Tuple>::value > 0,
            "Can't convert a zero-length std::tuple to a PyTuple");
        
        return py::impl::apply(py::tuplize<Args...>,
                               std::forward<Tuple>(argtuple));
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
                    return py::tuplize(image.dim(0));
                case 2:
                    return py::tuplize(image.dim(0),
                                       image.dim(1));
                case 3:
                    return py::tuplize(image.dim(0),
                                       image.dim(1),
                                       image.dim(2));
                case 4:
                    return py::tuplize(image.dim(0),
                                       image.dim(1),
                                       image.dim(2),
                                       image.dim(3));
                case 5:
                    return py::tuplize(image.dim(0),
                                       image.dim(1),
                                       image.dim(2),
                                       image.dim(3),
                                       image.dim(4));
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        template <typename ImageType> inline
        PyObject* image_strides(ImageType const& image) {
            switch (image.ndims()) {
                case 1:
                    return py::tuplize(image.stride(0));
                case 2:
                    return py::tuplize(image.stride(0),
                                       image.stride(1));
                case 3:
                    return py::tuplize(image.stride(0),
                                       image.stride(1),
                                       image.stride(2));
                case 4:
                    return py::tuplize(image.stride(0),
                                       image.stride(1),
                                       image.stride(2),
                                       image.stride(3));
                case 5:
                    return py::tuplize(image.stride(0),
                                       image.stride(1),
                                       image.stride(2),
                                       image.stride(3),
                                       image.stride(4));
                default:
                    return py::tuplize();
            }
            return py::tuplize();
        }
        
        /// A misnomer -- actually returns a dtype-compatible tuple full
        /// of label/format subtuples germane to the description
        /// of the parsed typecode you pass it
        PyObject* structcode_to_dtype(char const* code);
        
        /// One-and-done functions for dumping a tuple of python strings
        /// filled with valid (as in registered) format suffixes; currently
        /// we just call this the once in the extensions' init func and stash
        /// the return value in the `im` module PyObject, just in there like
        /// alongside the types and other junk
        // using stringvec_t = std::vector<std::string>;
        // stringvec_t& formats_as_vector();                   /// this one is GIL-optional (how european!)
        PyObject* formats_as_pytuple(int idx = 0);          /// whereas here, no GIL no shoes no funcall
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_