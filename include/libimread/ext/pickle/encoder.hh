/// Copyright 2014-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_PICKLE_ENCODER_HH_
#define LIBIMREAD_EXT_PICKLE_ENCODER_HH_

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <type_traits>
#include <initializer_list>

namespace store {
    
    /// forward declarations:
    class stringmapper;
    class stringmap;
    
    namespace pickle {
        
        namespace impl {
            
            __attribute__((format(printf, 1, 2)))
            void argcheck(const char *format, ...);
            va_list&& argcompand(std::nullptr_t, ...);
            
            #define static_argcheck(disambiguate, ...) \
                if (false) { impl::argcheck(__VA_ARGS__); }
            
            template <typename Tuple>
            std::size_t tuple_size_v = std::tuple_size<Tuple>::value;
            
            template <typename F, typename Tuple, std::size_t ...I> inline
            auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
                return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
            }
            
            template <typename F, typename Tuple> inline
            auto apply(F&& f, Tuple&& t) {
                using Indices = std::make_index_sequence<tuple_size_v<std::decay_t<Tuple>>>;
                return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices());
            }
        
        } /// namespace impl
        
        class encoder {
            
            protected:
                /// I/O-related API:
                virtual emit(char) = 0;
                virtual emit(char const*) = 0;
                virtual emit(std::string const&) = 0;
            
            private:
                /// declare the opcode emitter methods:
                #define OPCODE(__name__) void op_##__name__()
                OPCODE(MARK);
                OPCODE(STOP);
                OPCODE(POP);
                OPCODE(POP_MARK);
                OPCODE(DUP);
                OPCODE(FLOAT);
                OPCODE(INT);
                OPCODE(BININT);
                OPCODE(BININT1);
                OPCODE(LONG);
                OPCODE(BININT2);
                OPCODE(NONE);
                OPCODE(PERSID);
                OPCODE(BINPERSID);
                OPCODE(REDUCE);
                OPCODE(STRING);
                OPCODE(BINSTRING);
                OPCODE(SHORT_BINSTRING);
                OPCODE(UNICODE);
                OPCODE(BINUNICODE);
                OPCODE(APPEND);
                OPCODE(BUILD);
                OPCODE(GLOBAL);
                OPCODE(DICT);
                OPCODE(EMPTY_DICT);
                OPCODE(APPENDS);
                OPCODE(GET);
                OPCODE(BINGET);
                OPCODE(INST);
                OPCODE(LONG_BINGET);
                OPCODE(LIST);
                OPCODE(EMPTY_LIST);
                OPCODE(OBJ);
                OPCODE(PUT);
                OPCODE(BINPUT);
                OPCODE(LONG_BINPUT);
                OPCODE(SETITEM);
                OPCODE(TUPLE);
                OPCODE(EMPTY_TUPLE);
                OPCODE(SETITEMS);
                OPCODE(BINFLOAT);
                OPCODE(PROTO);
                OPCODE(NEWOBJ);
                OPCODE(EXT1);
                OPCODE(EXT2);
                OPCODE(EXT4);
                OPCODE(TUPLE1);
                OPCODE(TUPLE2);
                OPCODE(TUPLE3);
                OPCODE(NEWTRUE);
                OPCODE(NEWFALSE);
                OPCODE(LONG1);
                OPCODE(LONG4);
                OPCODE(BINBYTES);
                OPCODE(SHORT_BINBYTES);
                OPCODE(SHORT_BINUNICODE);
                OPCODE(BINUNICODE8);
                OPCODE(BINBYTES8);
                OPCODE(EMPTY_SET);
                OPCODE(ADDITEMS);
                OPCODE(FROZENSET);
                OPCODE(NEWOBJ_EX);
                OPCODE(STACK_GLOBAL);
                OPCODE(MEMOIZE);
                OPCODE(FRAME);
                #undef OPCODE
            
            private:
                void newline(); /// emit a newline (\n)
                void zero();    /// emit a zero (0)
                void one();     /// emit a one (1)
            
            public:
                /// void encode(…):
                void encode(std::nullptr_t);
                void encode(void);
                void encode(void*);
                void encode(bool);
                void encode(std::size_t);
                void encode(ssize_t);
                void encode(int8_t);
                void encode(int16_t);
                void encode(int32_t);
                void encode(int64_t);
                void encode(uint8_t);
                void encode(uint16_t);
                void encode(uint32_t);
                void encode(uint64_t);
                void encode(float);
                void encode(double);
                void encode(long double);
                void encode(char*);
                void encode(char const*);
                void encode(std::string const&);
                void encode(char*, std::size_t);
                void encode(char const*, std::size_t);
                void encode(std::string const&, std::size_t);
                void encode(std::wstring const&);
                void encode(std::wstring const&, std::size_t);
                
                /* >>>>>>>>>>>>>>>>>>> FORWARD DECLARATIONS <<<<<<<<<<<<<<<<<<<< */
                
                template <typename First, typename Second>
                void encode(std::pair<First, Second> pair);
                
                template <typename ...Args>
                void encode(std::tuple<Args...> argtuple);
                
                template <typename ...Args>
                void encode(std::tuple<Args&&...> argtuple);
                
                template <typename Vector,
                          typename Value = typename Vector::value_type>
                void encode(Vector const& vector);
                          
                template <typename Mapping,
                          typename Value = typename Mapping::mapped_type,
                          typename std::enable_if_t<
                                   std::is_constructible<std::string,
                                                         typename Mapping::key_type>::value,
                          int> = 0>
                void encode(Mapping const& mapping);
                
                /* >>>>>>>>>>>>>>>>>>> EXPLICIT CASTS <<<<<<<<<<<<<<<<<<<< */
                
                template <typename Cast,
                          typename Original,
                          typename std::enable_if_t<std::is_arithmetic<Cast>::value &&
                                                    std::is_arithmetic<Original>::value,
                          int> = 0> inline
                void encode(Original orig) {
                    store::pickle::encode(static_cast<Cast>(orig));
                }
                
                template <typename Cast,
                          typename Original,
                          typename std::enable_if_t<!std::is_arithmetic<Cast>::value &&
                                                    !std::is_arithmetic<Original>::value,
                          int> = 0> inline
                void encode(Original orig) {
                    store::pickle::encode(reinterpret_cast<Cast>(orig));
                }
                
                template <typename ...Args>
                using convertible = std::is_same<void,
                                    std::common_type_t<
                                        decltype(
                                        store::pickle::encode(std::declval<Args>()))...>>;
                                                 
                template <typename ...Args>
                bool convertible_v = store::pickle::convertible<Args...>::value;
                
                /* >>>>>>>>>>>>>>>>>>> PRINTF-ISH FORMAT STYLE <<<<<<<<<<<<<<<<<<<< */
                
                template <typename ...Args>
                __attribute__((nonnull(1)))
                void encode(char const* formatstring, bool disambiguate, Args&&... args) {
                    /// trigger compile-time type checks for printf-style variadics
                    /// -- this is a no-op at runtime:
                    static_argcheck(disambiguate,
                                    formatstring,
                                    std::forward<Args>(args)...);
                    
                    /// dispatch using argument pack companded as a va_list:
                    return store::pickle::encode(formatstring,
                           store::pickle::impl::argcompand(nullptr, std::forward<Args>(args)...));
                }
                
                /* >>>>>>>>>>>>>>>>>>> TUPLIZE <<<<<<<<<<<<<<<<<<<< */
                
                void tuplize();
                
                template <typename ListType,
                          typename std::enable_if_t<
                                    store::pickle::convertible<ListType>::value,
                          int> = 0>
                void tuplize(std::initializer_list<ListType> list) {
                    // Py_ssize_t idx = 0,
                    //            max = list.size();
                    // void tuple = PyTuple_New(max);
                    // for (ListType const& item : list) {
                    //     PyTuple_SET_ITEM(tuple, idx, store::pickle::encode(item));
                    //     idx++;
                    // }
                    // return tuple;
                }
                
                template <typename Args,
                          typename std::enable_if_t<
                                    store::pickle::convertible<Args>::value,
                          int> = 0>
                void tuplize(Args arg) {
                    // return PyTuple_Pack(1,
                    //     py::encode(std::forward<Args>(arg)));
                }
                
                template <typename ...Args>
                void tuplize(Args&& ...args) {
                    // static_assert(
                    //     sizeof...(Args) > 1,
                    //     "Can't tuplize a zero-length arglist");
                    //
                    // return PyTuple_Pack(
                    //     sizeof...(Args),
                    //     store::pickle::encode(std::forward<Args>(args))...);
                }
                
                /* >>>>>>>>>>>>>>>>>>> LISTIFY <<<<<<<<<<<<<<<<<<<< */
                
                void listify();
                
                template <typename ListType,
                          typename std::enable_if_t<
                                    store::pickle::convertible<ListType>::value,
                          int> = 0>
                void listify(std::initializer_list<ListType> list) {
                    // Py_ssize_t idx = 0,
                    //            max = list.size();
                    // void pylist = PyList_New(max);
                    // for (ListType const& item : list) {
                    //     PyList_SET_ITEM(pylist, idx, store::pickle::encode(item));
                    //     idx++;
                    // }
                    // return pylist;
                }
                
                template <typename Args,
                          typename std::enable_if_t<
                                    store::pickle::convertible<Args>::value,
                          int> = 0>
                void listify(Args arg) {
                    // void pylist = PyList_New(1);
                    // PyList_SET_ITEM(pylist, 0,
                    //     store::pickle::encode(std::forward<Args>(arg)));
                    // return pylist;
                }
                
                template <typename Tuple, std::size_t ...I>
                void listify(Tuple&& t, std::index_sequence<I...>) {
                    // void pylist = PyList_New(
                    //     std::tuple_size<Tuple>::value);
                    // unpack {
                    //     PyList_SET_ITEM(pylist, I,
                    //         store::pickle::encode(std::get<I>(std::forward<Tuple>(t))))...
                    // };
                    // return pylist;
                }
                
                template <typename ...Args>
                void listify(Args&& ...args) {
                    // using Indices = std::index_sequence_for<Args...>;
                    // static_assert(
                    //     sizeof...(Args) > 1,
                    //     "Can't listify a zero-length arglist");
                    //
                    // return store::pickle::listify(
                    //     std::forward_as_tuple(args...),
                    //     Indices());
                }
                
                /* >>>>>>>>>>>>>>>>>>> PAIR AND TUPLE CONVERTERS <<<<<<<<<<<<<<<<<<<< */
                
                template <typename First, typename Second>
                void encode(std::pair<First, Second> pair) {
                }
                
                template <typename ...Args>
                void encode(std::tuple<Args...> argtuple) {
                }
                
                template <typename ...Args>
                void encode(std::tuple<Args&&...> argtuple) {
                }
                
                /* >>>>>>>>>>>>>>>>>>> GENERIC DECONTAINERIZERS <<<<<<<<<<<<<<<<<<<< */
                
                template <typename Vector, typename Value>
                void encode(Vector const& vector) {
                }
                
                template <typename Mapping, typename Value,
                          typename std::enable_if_t<
                                   std::is_constructible<std::string,
                                                         typename Mapping::key_type>::value,
                          int>>
                void encode(Mapping const& mapping) {
                }
                
        };
        
        
        
        
    } /// namespace pickle
    
} /// namespace store

#endif /// LIBIMREAD_EXT_PICKLE_ENCODER_HH_