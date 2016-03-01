#if ! defined(_SG14_FIXED_POINT)
#define _SG14_FIXED_POINT 1

#include <climits>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <ostream>
#include <istream>
#include <iostream>
#include <functional>

// sg14::float_point only fully supports 64-bit types with the help of 128-bit ints.
// Clang and GCC use (__int128_t) and (unsigned __int128_t) for 128-bit ints.

#if defined(__clang__) || defined(__GNUG__)
#define _SG14_FIXED_POINT_128

namespace std {
    
    ////////////////////////////////////////////////////////////////////////////////
    // placeholder hashers for 128-bit integer types
    
    template <>
    struct hash<__int128_t> {
        
        typedef __int128_t argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& bigint) const {
            std::hash<int64_t> hasher;
            return static_cast<result_type>(
                hasher(static_cast<int64_t>(bigint)));
        }
        
    };
    
    template <>
    struct hash<__uint128_t> {
        
        typedef __uint128_t argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& biguint) const {
            std::hash<uint64_t> hasher;
            return static_cast<result_type>(
                hasher(static_cast<uint64_t>(biguint)));
        }
        
    };
    
} /* namespace std */

#endif

namespace sg14 {
    
    namespace impl {
        
        ////////////////////////////////////////////////////////////////////////////////
        // num_bits
        
        template <typename T>
        constexpr int num_bits()
        {
            return sizeof(T) * CHAR_BIT;
        }
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::get_int_t
        
        template <bool SIGNED, int NUM_BYTES>
        struct get_int;
        
        template <bool SIGNED, int NUM_BYTES>
        using get_int_t = typename get_int<SIGNED, NUM_BYTES>::type;
        
        // specializations
        template <> struct get_int<false, 1>    { using type = std::uint8_t; };
        template <> struct get_int<true, 1>     { using type = std::int8_t; };
        template <> struct get_int<false, 2>    { using type = std::uint16_t; };
        template <> struct get_int<true, 2>     { using type = std::int16_t; };
        template <> struct get_int<false, 4>    { using type = std::uint32_t; };
        template <> struct get_int<true, 4>     { using type = std::int32_t; };
        template <> struct get_int<false, 8>    { using type = std::uint64_t; };
        template <> struct get_int<true, 8>     { using type = std::int64_t; };
        #if defined(_SG14_FIXED_POINT_128)
        template <> struct get_int<false, 16>   { using type = __uint128_t; };
        template <> struct get_int<true, 16>    { using type = __int128_t; };
        #endif
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::is_integral
        
        template <typename T>
        struct is_integral;
        
        // possible exception to std::is_integral as fixed_point<bool, X> seems pointless
        template <>
        struct is_integral<bool> : std::false_type {};
        
        #if defined(_SG14_FIXED_POINT_128)
        // addresses https://llvm.org/bugs/show_bug.cgi?id=23156
        template <>
        struct is_integral<__int128_t> : std::true_type {};
        
        template <>
        struct is_integral<__uint128_t> : std::true_type {};
        #endif
        
        template <typename T>
        struct is_integral : std::is_integral<T> {};
        
        template <typename T>
        struct is_floating_point : std::is_floating_point<T> {};
        
        template <typename T, typename U>
        static constexpr bool is_same_v = std::is_same<T, U>::value;
        
        template <typename T>
        static constexpr bool is_integral_v = is_integral<T>::value;
        
        template <typename T>
        static constexpr bool is_floating_point_v = is_floating_point<T>::value;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::is_signed
        
        template <typename T>
        struct is_signed
        {
            static_assert(is_integral_v<T>,
                          "sg14::impl::is_signed only intended for use with integral types");
            static constexpr bool value = impl::is_same_v<get_int_t<true, sizeof(T)>, T>;
        };
        
        template <typename T>
        static constexpr bool is_signed_v = is_signed<T>::value;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::is_unsigned
        
        template <typename T>
        struct is_unsigned
        {
            static_assert(is_integral_v<T>,
                          "sg14::impl::is_unsigned only intended for use with integral types");
            static constexpr bool value = impl::is_same_v<get_int_t<false, sizeof(T)>, T>;
        };
        
        template <typename T>
        static constexpr bool is_unsigned_v = is_unsigned<T>::value;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::make_signed
        
        template <typename T>
        struct make_signed
        {
            using type = get_int_t<true, sizeof(T)>;
        };
        
        template <typename T>
        using make_signed_t = typename make_signed<T>::type;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::make_unsigned
        
        template <typename T>
        struct make_unsigned
        {
            using type = get_int_t<false, sizeof(T)>;
        };
        
        template <typename T>
        using make_unsigned_t = typename make_unsigned<T>::type;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::next_size_t
        
        // given an integral type, INT_TYPE,
        // provides the integral type of the equivalent type with twice the size
        template <typename INT_TYPE>
        using next_size_t = get_int_t<
            impl::is_signed_v<INT_TYPE>,
            sizeof(INT_TYPE) * 2>;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::previous_size_t
        
        // given an integral type, INT_TYPE,
        // provides the integral type of the equivalent type with half the size
        template <typename INT_TYPE>
        using previous_size_t = get_int_t<
            impl::is_signed_v<INT_TYPE>,
            sizeof(INT_TYPE) / 2>;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::shift_left and sg14::impl::shift_right
        
        // performs a shift operation by a fixed number of bits avoiding two pitfalls:
        // 1) shifting by a negative amount causes undefined behavior
        // 2) converting between integer types of different sizes can lose significant bits during shift right
        
        // EXPONENT >= 0 && sizeof(OUTPUT) <= sizeof(INPUT) && is_unsigned<INPUT>
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT >= 0 && sizeof(OUTPUT) <= sizeof(INPUT) &&
                           impl::is_unsigned_v<INPUT>, int> dummy = 0>
        constexpr OUTPUT shift_left(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            return static_cast<OUTPUT>(i) << EXPONENT;
        }
        
        // EXPONENT >= 0 && sizeof(OUTPUT) <= sizeof(INPUT) && is_signed<INPUT>
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT >= 0 && sizeof(OUTPUT) <= sizeof(INPUT) &&
                           impl::is_signed_v<INPUT>, int> dummy = 0>
        constexpr OUTPUT shift_left(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            using signed_type = impl::make_signed_t<OUTPUT>;
            return (i >= 0) ?
                static_cast<OUTPUT>(i) << EXPONENT :
                static_cast<OUTPUT>(-(static_cast<signed_type>(-i) << EXPONENT));
        }
        
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT >= 0 && sizeof(OUTPUT) <= sizeof(INPUT), int> dummy = 0>
        constexpr OUTPUT shift_right(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            return static_cast<OUTPUT>(i >> EXPONENT);
        }
        
        // EXPONENT >= 0 && sizeof(OUTPUT) > sizeof(INPUT) && is_unsigned<INPUT>
        template <int EXPONENT, 
                  typename OUTPUT, 
                  typename INPUT, 
                  typename std::enable_if_t<EXPONENT >= 0 && !(sizeof(OUTPUT) <= sizeof(INPUT)) &&
                           impl::is_unsigned_v<INPUT>, char> dummy = 0>
        constexpr OUTPUT shift_left(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            return static_cast<OUTPUT>(i) << EXPONENT;
        }
        
        // EXPONENT >= 0 && sizeof(OUTPUT) > sizeof(INPUT) && is_signed<INPUT>
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT >= 0 && !(sizeof(OUTPUT) <= sizeof(INPUT)) &&
                           impl::is_signed_v<INPUT>, char> dummy = 0>
        constexpr OUTPUT shift_left(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            using signed_type = typename impl::make_signed_t<OUTPUT>;
            return (i >= 0)
                ? static_cast<OUTPUT>(i) << EXPONENT
                : static_cast<OUTPUT>(-(static_cast<signed_type>(-i) << EXPONENT));
        }
        
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT >= 0 && !(sizeof(OUTPUT) <= sizeof(INPUT)), char> dummy = 0>
        constexpr OUTPUT shift_right(INPUT i) noexcept
        {
            static_assert(impl::is_integral_v<INPUT>,
                          "INPUT must be integral type");
            static_assert(impl::is_integral_v<OUTPUT>,
                          "OUTPUT must be integral type");
            return static_cast<OUTPUT>(i) >> EXPONENT;
        }
        
        // pass bit-shifts with negative EXPONENTS to their complimentary positive-EXPONENT equivalents
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<(EXPONENT < 0), int> dummy = 0>
        constexpr OUTPUT shift_left(INPUT i) noexcept
        {
            return shift_right<-EXPONENT, OUTPUT, INPUT>(i);
        }
        
        template <int EXPONENT,
                  typename OUTPUT,
                  typename INPUT,
                  typename std::enable_if_t<EXPONENT < 0, int> dummy = 0>
        constexpr OUTPUT shift_right(INPUT i) noexcept
        {
            return shift_left<-EXPONENT, OUTPUT, INPUT>(i);
        }
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::default_exponent
        
        template <typename REPR_TYPE>
        constexpr int default_exponent() noexcept
        {
            return num_bits<REPR_TYPE>() / -2;
        }
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::pow2
        
        // returns given power of 2
        template <typename S, int EXPONENT,
                  typename std::enable_if_t<EXPONENT == 0, int> dummy = 0>
        constexpr S pow2() noexcept
        {
            static_assert(impl::is_floating_point_v<S>,
                          "S must be floating-point type");
            return 1;
        }
        
        template <typename S, int EXPONENT,
                  typename std::enable_if_t<!(EXPONENT <= 0), int> dummy = 0>
        constexpr S pow2() noexcept
        {
            static_assert(impl::is_floating_point_v<S>,
                          "S must be floating-point type");
            return pow2<S, EXPONENT - 1>() * S(2);
        }
        
        template <typename S, int EXPONENT,
                  typename std::enable_if_t<!(EXPONENT >= 0), int> dummy = 0>
        constexpr S pow2() noexcept
        {
            static_assert(impl::is_floating_point_v<S>,
                          "S must be floating-point type");
            return pow2<S, EXPONENT + 1>() * S(.5);
        }
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::impl::capacity
        
        // has value that, given a value N, 
        // returns number of bits necessary to represent it in binary
        template <unsigned N> struct capacity;
        
        template <>
        struct capacity<0> {
            static constexpr int value = 0;
        };
        
        template <unsigned N>
        struct capacity {
            static constexpr int value = capacity<N / 2>::value + 1;
        };
        
        template <unsigned N>
        static constexpr int capacity_v = capacity<N>::value;
        
        ////////////////////////////////////////////////////////////////////////////////
        // impl::necessary_repr_t
        
        // given a required number of bits a type should have and whether it is signed,
        // provides a built-in integral type with necessary capacity
        template <unsigned REQUIRED_BITS,
                  bool IS_SIGNED>
        using necessary_repr_t 
            = get_int_t<IS_SIGNED, 1 << (capacity_v<((REQUIRED_BITS + 7) / 8) - 1>)>;
        
        ////////////////////////////////////////////////////////////////////////////////
        // sg14::sqrt helper functions
        
        template <typename REPR_TYPE>
        constexpr REPR_TYPE sqrt_bit(
            REPR_TYPE n,
            REPR_TYPE bit = REPR_TYPE(1) << (num_bits<REPR_TYPE>() - 2)) noexcept
        {
            return (bit > n) ? sqrt_bit<REPR_TYPE>(n, bit >> 2) : bit;
        }
        
        template <typename REPR_TYPE>
        constexpr REPR_TYPE sqrt_solve3(
            REPR_TYPE n,
            REPR_TYPE bit,
            REPR_TYPE result) noexcept
        {
            return bit
                   ? (n >= result + bit)
                     ? sqrt_solve3<REPR_TYPE>(n - (result + bit), bit >> 2, (result >> 1) + bit)
                     : sqrt_solve3<REPR_TYPE>(n, bit >> 2, result >> 1)
                   : result;
        }
        
        template <typename REPR_TYPE>
        constexpr REPR_TYPE sqrt_solve1(REPR_TYPE n) noexcept
        {
            return sqrt_solve3<REPR_TYPE>(n, sqrt_bit<REPR_TYPE>(n), 0);
        }
        
    } /* namespace impl */
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point class template definition
    //
    // approximates a real number using a built-in integral type;
    // somewhat like a floating-point number but - with exponent determined at run-time
    
    template <typename REPR_TYPE,
              int EXPONENT = impl::default_exponent<REPR_TYPE>()>
    class fixed_point {
        
        public:
            ////////////////////////////////////////////////////////////////////////////////
            // types
            
            using repr_type = REPR_TYPE;
            
            ////////////////////////////////////////////////////////////////////////////////
            // constants
            
            constexpr static int exponent = EXPONENT;
            constexpr static int digits = impl::num_bits<REPR_TYPE>() - impl::is_signed_v<repr_type>;
            constexpr static int integer_digits = digits + exponent;
            constexpr static int fractional_digits = digits - integer_digits;
            
            ////////////////////////////////////////////////////////////////////////////////
            // functions
            
        private:
            // constructor taking representation explicitly using operator++(int)-style trick
            explicit constexpr fixed_point(repr_type repr, int) noexcept
                : repr_value(repr)
                {}
            
        public:
            // default c'tor
            // fixed_point() noexcept {}
            explicit constexpr fixed_point() noexcept {}
            
            // c'tor taking an integer type
            template <typename S,
                      typename std::enable_if_t<impl::is_integral_v<S>, int> dummy = 0>
            explicit constexpr fixed_point(S s) noexcept
                : repr_value(integral_to_repr(s))
                {}
            
            // c'tor taking a floating-point type
            template <typename S,
                      typename std::enable_if_t<impl::is_floating_point_v<S>, int> dummy = 0>
            explicit constexpr fixed_point(S s) noexcept
                : repr_value(floating_point_to_repr(s))
                {}
            
            // c'tor taking a fixed-point type
            template <typename FROM_REPR_TYPE,
                      int FROM_EXPONENT>
            explicit constexpr fixed_point(fixed_point<FROM_REPR_TYPE, FROM_EXPONENT> const& rhs) noexcept
                : repr_value(impl::shift_right<(exponent - FROM_EXPONENT), repr_type>(rhs.data()))
                {}
            
            constexpr fixed_point(fixed_point const& rhs) noexcept
                : repr_value(rhs.repr_value)
                {}
            
            constexpr fixed_point(fixed_point&& rhs) noexcept
                : repr_value(rhs.repr_value)
                {}
            
            template <typename FROM_REPR_TYPE,
                      int FROM_EXPONENT>
            constexpr fixed_point& operator=(fixed_point<FROM_REPR_TYPE, FROM_EXPONENT>&& rhs) noexcept
            {
                repr_value = impl::shift_right<(exponent - FROM_EXPONENT), repr_type>(rhs.data());
                return *this;
            }
            
            template <typename S,
                      typename std::enable_if_t<impl::is_integral_v<S>, int> dummy = 0>
            constexpr fixed_point& operator=(S s) noexcept
            {
                repr_value = integral_to_repr(s);
                return *this;
            }
            
            template <typename S,
                      typename std::enable_if_t<impl::is_floating_point_v<S>, int> dummy = 0>
            constexpr fixed_point& operator=(S s) noexcept
            {
                repr_value = floating_point_to_repr(s);
                return *this;
            }
            
            constexpr fixed_point& operator=(fixed_point const& rhs) noexcept
            {
                repr_value = rhs.repr_value;
                return *this;
            }
            
            constexpr fixed_point& operator=(fixed_point&& rhs) noexcept
            {
                repr_value = rhs.repr_value;
                return *this;
            }
            
            // returns value represented as a floating-point
            template <typename S,
                      typename std::enable_if_t<impl::is_integral_v<S>, int> dummy = 0>
            explicit constexpr operator S() const noexcept
            {
                return repr_to_integral<S>(repr_value);
            }
            
            // returns value represented as integral
            template <typename S,
                      typename std::enable_if_t<impl::is_floating_point_v<S>, int> dummy = 0>
            explicit constexpr operator S() const noexcept
            {
                return repr_to_floating_point<S>(repr_value);
            }
            
            // returns internal representation of value
            constexpr repr_type data() const noexcept
            {
                return repr_value;
            }
            
            // creates an instance given the underlying representation value
            // TODO: constexpr with c++14?
            static constexpr fixed_point from_data(repr_type repr) noexcept
            {
                return fixed_point(repr, 0);
            }
            
            // comparison
            friend constexpr bool operator==(fixed_point const& lhs,
                                             fixed_point const& rhs) noexcept
            {
                return lhs.repr_value == rhs.repr_value;
            }
            friend constexpr bool operator!=(fixed_point const& lhs,
                                             fixed_point const& rhs) noexcept
            {
                return !(lhs == rhs);
            }
            
            friend constexpr bool operator>(fixed_point const& lhs,
                                            fixed_point const& rhs) noexcept
            {
                return lhs.repr_value > rhs.repr_value;
            }
            friend constexpr bool operator<(fixed_point const& lhs,
                                            fixed_point const& rhs) noexcept
            {
                return lhs.repr_value < rhs.repr_value;
            }
            
            friend constexpr bool operator>=(fixed_point const& lhs,
                                             fixed_point const& rhs) noexcept
            {
                return lhs.repr_value >= rhs.repr_value;
            }
            friend constexpr bool operator<=(fixed_point const& lhs,
                                             fixed_point const& rhs) noexcept
            {
                return lhs.repr_value <= rhs.repr_value;
            }
            
            // arithmetic
            friend constexpr fixed_point operator-(fixed_point const& rhs) noexcept
            {
                static_assert(impl::is_signed_v<repr_type>,
                              "unary negation of unsigned value");
                return fixed_point(-rhs.repr_value, 0);
            }
            
            friend constexpr fixed_point operator+(fixed_point const& lhs,
                                                   fixed_point const& rhs) noexcept
            {
                return fixed_point(lhs.repr_value + rhs.repr_value, 0);
            }
            friend constexpr fixed_point operator-(fixed_point const& lhs,
                                                   fixed_point const& rhs) noexcept
            {
                return fixed_point(lhs.repr_value - rhs.repr_value, 0);
            }
            friend constexpr fixed_point operator*(fixed_point const& lhs,
                                                   fixed_point const& rhs) noexcept
            {
                return fixed_point(impl::shift_left<exponent, repr_type>(
                    impl::next_size_t<repr_type>(
                        lhs.repr_value) * rhs.repr_value), 0);
            }
            friend constexpr fixed_point operator/(fixed_point const& lhs,
                                                   fixed_point const& rhs) noexcept
            {
                return fixed_point(repr_type(
                    impl::shift_right<exponent, impl::next_size_t<repr_type>>(
                        lhs.repr_value) / rhs.repr_value), 0);
            }
            
            friend fixed_point& operator+=(fixed_point& lhs,
                                           fixed_point const& rhs) noexcept
            {
                return lhs = lhs + rhs;
            }
            friend fixed_point& operator-=(fixed_point& lhs,
                                           fixed_point const& rhs) noexcept
            {
                return lhs = lhs - rhs;
            }
            friend fixed_point& operator*=(fixed_point& lhs,
                                           fixed_point const& rhs) noexcept
            {
                return lhs = lhs * rhs;
            }
            friend fixed_point& operator/=(fixed_point& lhs,
                                           fixed_point const& rhs) noexcept
            {
                return lhs = lhs / rhs;
            }
            
        private:
            template <typename S,
                      typename std::enable_if_t<impl::is_floating_point_v<S>, int> dummy = 0>
            static constexpr S one() noexcept
            {
                return impl::pow2<S, -exponent>();
            }
            
            template <typename S,
                      typename std::enable_if_t<impl::is_integral_v<S>, int> dummy = 0>
            static constexpr S one() noexcept
            {
                return integral_to_repr<S>(1);
            }
            
            template <typename S>
            static constexpr S inverse_one() noexcept
            {
                static_assert(impl::is_floating_point_v<S>,
                              "S must be floating-point type");
                return impl::pow2<S, exponent>();
            }
            
            template <typename S>
            static constexpr repr_type integral_to_repr(S s) noexcept
            {
                static_assert(impl::is_integral_v<S>,
                              "S must be unsigned integral type");
                return impl::shift_right<exponent, repr_type>(s);
            }
            
            template <typename S>
            static constexpr S repr_to_integral(repr_type r) noexcept
            {
                static_assert(impl::is_integral_v<S>,
                              "S must be unsigned integral type");
                return impl::shift_left<exponent, S>(r);
            }
            
            template <typename S>
            static constexpr repr_type floating_point_to_repr(S s) noexcept
            {
                static_assert(impl::is_floating_point_v<S>,
                              "S must be floating-point type");
                return static_cast<repr_type>(s * one<S>());
            }
            
            template <typename S>
            static constexpr S repr_to_floating_point(repr_type r) noexcept
            {
                static_assert(impl::is_floating_point_v<S>,
                              "S must be floating-point type");
                return S(r) * inverse_one<S>();
            }
        
        public:
            void swap(fixed_point& rhs) noexcept
            {
                using std::swap;
                swap(repr_value, rhs.repr_value);
            }
            
            friend void swap(fixed_point& lhs, fixed_point& rhs) noexcept
            {
                using std::swap;
                swap(lhs.repr_value, rhs.repr_value);
            }
            
            std::size_t hash() const noexcept
            {
                std::hash<repr_type> hasher;
                return hasher(repr_value);
            }
        
        private:
            ////////////////////////////////////////////////////////////////////////////////
            // variables
            
            repr_type repr_value = 0;
    };
    
    template <typename REPR_TYPE,
              int EXPONENT = impl::default_exponent<REPR_TYPE>()>
    union alignas(REPR_TYPE) fixed_repr_t {
        using fixed_t = fixed_point<REPR_TYPE, EXPONENT>;
        fixed_t value = fixed_t(0);
        REPR_TYPE repr;
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::make_fixed
    
    // given the desired number of integer and fractional digits,
    // generates a fixed_point type such that:
    //   fixed_point<>::integer_digits == INTEGER_DIGITS,
    // and
    //   fixed_point<>::fractional_digits >= FRACTIONAL_DIGITS,
    template <unsigned INTEGER_DIGITS,
              unsigned FRACTIONAL_DIGITS,
              bool IS_SIGNED = true>
    using make_fixed = fixed_point<
        typename impl::necessary_repr_t<INTEGER_DIGITS + FRACTIONAL_DIGITS + IS_SIGNED, IS_SIGNED>,
        (signed)(INTEGER_DIGITS + IS_SIGNED) -
            impl::num_bits<typename impl::necessary_repr_t<INTEGER_DIGITS + FRACTIONAL_DIGITS + IS_SIGNED, IS_SIGNED>>()>;
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::make_fixed_from_repr
    
    // yields a float_point with EXPONENT calculated such that 
    // fixed_point<REPR_TYPE, EXPONENT>::integer_bits == INTEGER_BITS
    template <typename REPR_TYPE, int INTEGER_BITS>
    using make_fixed_from_repr = fixed_point<
        REPR_TYPE,
        INTEGER_BITS + impl::is_signed_v<REPR_TYPE> - (signed)sizeof(REPR_TYPE) * CHAR_BIT>;
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::open_unit and sg14::closed_unit partial specializations of fixed_point
    
    // fixed-point type capable of storing values in the range [0, 1);
    // a bit more precise than closed_unit
    template <typename REPR_TYPE>
    using open_unit = fixed_point<REPR_TYPE, -impl::num_bits<REPR_TYPE>()>;
    
    // fixed-point type capable of storing values in the range [0, 1];
    // actually storing values in the range [0, 2);
    // a bit less precise than closed_unit
    template <typename REPR_TYPE>
    using closed_unit = fixed_point<
        typename std::enable_if_t<impl::is_unsigned_v<REPR_TYPE>, REPR_TYPE>,
        1 - impl::num_bits<REPR_TYPE>()>;
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point_promotion_t / promote
    
    // given template parameters of a fixed_point specialization, 
    // yields alternative specialization with twice the fractional bits
    // and twice the integral/sign bits
    template <typename REPR_TYPE, int EXPONENT>
    using fixed_point_promotion_t = fixed_point<impl::next_size_t<REPR_TYPE>, EXPONENT * 2>;
    
    // as fixed_point_promotion_t but promotes parameter, from
    template <typename REPR_TYPE, int EXPONENT>
    constexpr fixed_point_promotion_t<REPR_TYPE, EXPONENT> promote(fixed_point<REPR_TYPE, EXPONENT> const& from) noexcept
    {
        return fixed_point_promotion_t<REPR_TYPE, EXPONENT>(from);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point_demotion_t / demote
    
    // given template parameters of a fixed_point specialization, 
    // yields alternative specialization with half the fractional bits
    // and half the integral/sign bits (assuming EXPONENT is even)
    template <typename REPR_TYPE, int EXPONENT>
    using fixed_point_demotion_t = fixed_point<impl::previous_size_t<REPR_TYPE>, EXPONENT / 2>;
    
    // as fixed_point_demotion_t but demotes parameter, from
    template <typename REPR_TYPE, int EXPONENT>
    constexpr fixed_point_demotion_t<REPR_TYPE, EXPONENT> demote(fixed_point<REPR_TYPE, EXPONENT> const& from) noexcept
    {
        return from;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point_mul_result_t / safe_multiply
    //
    // TODO: accept factors of heterogeneous specialization, e.g.:
    //       fixed_point<char, -4> * fixed_point<short, -10> = fixed_point<short, -7>
    
    // yields specialization of fixed_point with integral bits necessary to store 
    // result of a multiply between values of fixed_point<REPR_TYPE, EXPONENT>
    template <typename REPR_TYPE, int EXPONENT>
    using fixed_point_mul_result_t = make_fixed_from_repr<
        REPR_TYPE,
        fixed_point<REPR_TYPE, EXPONENT>::integer_digits * 2>;
    
    // as fixed_point_mul_result_t but converts parameter, factor,
    // ready for safe binary multiply
    template <typename REPR_TYPE, int EXPONENT>
    constexpr fixed_point_mul_result_t<REPR_TYPE, EXPONENT> safe_multiply(
        fixed_point<REPR_TYPE, EXPONENT> const& factor1,
        fixed_point<REPR_TYPE, EXPONENT> const& factor2) noexcept
    {
        using output_type = fixed_point_mul_result_t<REPR_TYPE, EXPONENT>;
        using next_type = impl::next_size_t<REPR_TYPE>;
        return output_type(
            impl::shift_left<EXPONENT * 2, next_type>(
                next_type(factor1.data()) * factor2.data()));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point_add_result_t / safe_add
    
    // yields specialization of fixed_point with integral bits necessary to store 
    // result of an addition between N values of fixed_point<REPR_TYPE, EXPONENT>
    template <typename REPR_TYPE, int EXPONENT,
              unsigned N = 2>
    using fixed_point_add_result_t = make_fixed_from_repr<
        REPR_TYPE,
        fixed_point<REPR_TYPE, EXPONENT>::integer_digits +
                                    impl::capacity_v<N - 1>>;
    
    namespace impl {
        
        template <typename RESULT_TYPE,
                  typename REPR_TYPE, int EXPONENT,
                  typename HEAD>
        constexpr RESULT_TYPE add(HEAD const& addend_head)
        {
            static_assert(impl::is_same_v<fixed_point<REPR_TYPE, EXPONENT>, HEAD>,
                          "mismatched safe_add parameters");
            return static_cast<RESULT_TYPE>(addend_head);
        }
        
        template <typename RESULT_TYPE,
                  typename REPR_TYPE, int EXPONENT,
                  typename HEAD,
                  typename ...TAIL>
        constexpr RESULT_TYPE add(HEAD const& addend_head, TAIL const& ...addend_tail)
        {
            static_assert(impl::is_same_v<fixed_point<REPR_TYPE, EXPONENT>, HEAD>,
                          "mismatched safe_add parameters");
            return add<RESULT_TYPE, REPR_TYPE, EXPONENT, TAIL...>(addend_tail...) +
                   static_cast<RESULT_TYPE>(addend_head);
        }
        
    }
    
    template <typename REPR_TYPE, int EXPONENT,
              typename ...TAIL>
    constexpr fixed_point_add_result_t<REPR_TYPE,
                                       EXPONENT,
                                       sizeof...(TAIL) + 1> safe_add(fixed_point<REPR_TYPE, EXPONENT> const& addend1,
                                                                     TAIL const& ...addend_tail)
    {
        using output_type = fixed_point_add_result_t<REPR_TYPE, EXPONENT, sizeof...(TAIL) + 1>;
        return impl::add<output_type, REPR_TYPE, EXPONENT>(addend1, addend_tail...);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::lerp
    
    // linear interpolation between two fixed_point values
    // given floating-point `t` for which result is `from` when t==0 and `to` when t==1
    template <typename REPR_TYPE, int EXPONENT,
              typename S>
    constexpr fixed_point<REPR_TYPE, EXPONENT> lerp(
        fixed_point<REPR_TYPE, EXPONENT> from,
        fixed_point<REPR_TYPE, EXPONENT> to,
        S t)
    {
        using closed_unit = closed_unit<typename impl::make_unsigned_t<REPR_TYPE>>;
        return lerp<REPR_TYPE, EXPONENT>(from, to, closed_unit(t));
    }
    
    template <typename REPR_TYPE, int EXPONENT>
    constexpr fixed_point<REPR_TYPE, EXPONENT> lerp(
        fixed_point<REPR_TYPE, EXPONENT> from,
        fixed_point<REPR_TYPE, EXPONENT> to,
        closed_unit<typename impl::make_unsigned_t<REPR_TYPE>> t)
    {
        using fixed_point = fixed_point<REPR_TYPE, EXPONENT>;
        using repr_type = typename fixed_point::repr_type;
        using next_repr_type = typename impl::next_size_t<repr_type>;
        using closed_unit = closed_unit<typename impl::make_unsigned_t<REPR_TYPE>>;
        return fixed_point::from_data(
            impl::shift_left<closed_unit::exponent, repr_type>(
                (static_cast<next_repr_type>(from.data()) * (closed_unit(1).data() - t.data())) +
                (static_cast<next_repr_type>(to.data()) * t.data())));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::abs
    
    template <typename REPR_TYPE, int EXPONENT,
              typename std::enable_if_t<impl::is_signed_v<REPR_TYPE>, int> dummy = 0>
    constexpr fixed_point<REPR_TYPE, EXPONENT> abs(fixed_point<REPR_TYPE, EXPONENT> const& x) noexcept
    {
        return (x.data() >= 0) ? x : - x;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::sqrt
    
    // https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Binary_numeral_system_.28base_2.29
    // slow when calculated at run-time?
    template <typename REPR_TYPE, int EXPONENT>
    constexpr fixed_point<REPR_TYPE, EXPONENT> sqrt(fixed_point<REPR_TYPE, EXPONENT> const& x) noexcept
    {
        return fixed_point<REPR_TYPE, EXPONENT>::from_data(
            static_cast<REPR_TYPE>(impl::sqrt_solve1(promote(x).data())));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // sg14::fixed_point streaming - (placeholder implementation)
    
    template <typename REPR_TYPE, int EXPONENT>
    std::ostream& operator<<(std::ostream& out, fixed_point<REPR_TYPE, EXPONENT> const& fp)
    {
        return out << static_cast<long double>(fp);
    }
    
    template <typename REPR_TYPE, int EXPONENT>
    std::ostream& operator<<(std::ostream& out, fixed_repr_t<REPR_TYPE, EXPONENT> const& fp)
    {
        return out << static_cast<long double>(fp.value);
    }
    
    template <typename REPR_TYPE, int EXPONENT>
    std::istream& operator>>(std::istream& in, fixed_point<REPR_TYPE, EXPONENT>& fp)
    {
        long double ld;
        in >> ld;
        fp = ld;
        return in;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // fixed_point specializations
    
    using fixed0_7_t    = fixed_point<std::int8_t, -7>;
    using fixed1_6_t    = fixed_point<std::int8_t, -6>;
    using fixed3_4_t    = fixed_point<std::int8_t, -4>;
    using fixed4_3_t    = fixed_point<std::int8_t, -3>;
    using fixed7_0_t    = fixed_point<std::int8_t,  0>;
    
    using ufixed0_8_t   = fixed_point<std::uint8_t, -8>;
    using ufixed1_7_t   = fixed_point<std::uint8_t, -7>;
    using ufixed4_4_t   = fixed_point<std::uint8_t, -4>;
    using ufixed8_0_t   = fixed_point<std::uint8_t,  0>;
    
    using fixed0_15_t   = fixed_point<std::int16_t, -15>;
    using fixed1_14_t   = fixed_point<std::int16_t, -14>;
    using fixed7_8_t    = fixed_point<std::int16_t, -8>;
    using fixed8_7_t    = fixed_point<std::int16_t, -7>;
    using fixed15_0_t   = fixed_point<std::int16_t,  0>;
    
    using ufixed0_16_t  = fixed_point<std::uint16_t, -16>;
    using ufixed1_15_t  = fixed_point<std::uint16_t, -15>;
    using ufixed8_8_t   = fixed_point<std::uint16_t, -8>; /// * u8Fixed8Number
    using ufixed16_0_t  = fixed_point<std::uint16_t,  0>;
    
    using fixed0_31_t   = fixed_point<std::int32_t, -31>;
    using fixed1_30_t   = fixed_point<std::int32_t, -30>;
    using fixed15_16_t  = fixed_point<std::int32_t, -16>; /// * s15Fixed16Number
    using fixed16_15_t  = fixed_point<std::int32_t, -15>;
    using fixed31_0_t   = fixed_point<std::int32_t,  0>;
    
    using ufixed0_32_t  = fixed_point<std::uint32_t, -32>;
    using ufixed1_31_t  = fixed_point<std::uint32_t, -31>;
    using ufixed16_16_t = fixed_point<std::uint32_t, -16>; /// * u16Fixed16Number
    using ufixed32_0_t  = fixed_point<std::uint32_t,  0>;
    
    using fixed0_63_t   = fixed_point<std::int64_t, -63>;
    using fixed1_62_t   = fixed_point<std::int64_t, -62>;
    using fixed31_32_t  = fixed_point<std::int64_t, -32>;
    using fixed32_31_t  = fixed_point<std::int64_t, -31>;
    using fixed63_0_t   = fixed_point<std::int64_t,  0>;
    
    using ufixed0_64_t  = fixed_point<std::uint64_t, -64>;
    using ufixed1_63_t  = fixed_point<std::uint64_t, -63>;
    using ufixed32_32_t = fixed_point<std::uint64_t, -32>;
    using ufixed64_0_t  = fixed_point<std::uint64_t,  0>;
    
    #if defined(_SG14_FIXED_POINT_128)
    using fixed0_127_t  = fixed_point<__int128_t, -127>;
    using fixed1_126_t  = fixed_point<__int128_t, -126>;
    using fixed63_64_t  = fixed_point<__int128_t, -64>;
    using fixed64_63_t  = fixed_point<__int128_t, -63>;
    using fixed127_0_t  = fixed_point<__int128_t,  0>;
    
    using ufixed0_128_t = fixed_point<__uint128_t, -128>;
    using ufixed1_127_t = fixed_point<__uint128_t, -127>;
    using ufixed64_64_t = fixed_point<__uint128_t, -64>;
    using ufixed128_0_t = fixed_point<__uint128_t,  0>;
    #endif
    
} /* namespace sg14 */

namespace std {
    
    #define DECLARE_HASHER(type)                                                \
    template <>                                                                 \
    struct hash<sg14::type> {                                                   \
        std::size_t operator()(sg14::type const& fixed_number) const {          \
            return static_cast<std::size_t>(fixed_number.hash());               \
        }                                                                       \
    };
    
    DECLARE_HASHER(fixed0_7_t);
    DECLARE_HASHER(fixed1_6_t);
    DECLARE_HASHER(fixed3_4_t);
    DECLARE_HASHER(fixed4_3_t);
    DECLARE_HASHER(fixed7_0_t);
    
    DECLARE_HASHER(ufixed0_8_t);
    DECLARE_HASHER(ufixed1_7_t);
    DECLARE_HASHER(ufixed4_4_t);
    DECLARE_HASHER(ufixed8_0_t);
    
    DECLARE_HASHER(fixed0_15_t);
    DECLARE_HASHER(fixed1_14_t);
    DECLARE_HASHER(fixed7_8_t);
    DECLARE_HASHER(fixed8_7_t);
    DECLARE_HASHER(fixed15_0_t);
    
    DECLARE_HASHER(ufixed0_16_t);
    DECLARE_HASHER(ufixed1_15_t);
    DECLARE_HASHER(ufixed8_8_t);
    DECLARE_HASHER(ufixed16_0_t);
    
    DECLARE_HASHER(fixed0_31_t);
    DECLARE_HASHER(fixed1_30_t);
    DECLARE_HASHER(fixed15_16_t);
    DECLARE_HASHER(fixed16_15_t);
    DECLARE_HASHER(fixed31_0_t);
    
    DECLARE_HASHER(ufixed0_32_t);
    DECLARE_HASHER(ufixed1_31_t);
    DECLARE_HASHER(ufixed16_16_t);
    DECLARE_HASHER(ufixed32_0_t);
    
    DECLARE_HASHER(fixed0_63_t);
    DECLARE_HASHER(fixed1_62_t);
    DECLARE_HASHER(fixed31_32_t);
    DECLARE_HASHER(fixed32_31_t);
    DECLARE_HASHER(fixed63_0_t);
    
    DECLARE_HASHER(ufixed0_64_t);
    DECLARE_HASHER(ufixed1_63_t);
    DECLARE_HASHER(ufixed32_32_t);
    DECLARE_HASHER(ufixed64_0_t);
    
    #if defined(_SG14_FIXED_POINT_128)
    DECLARE_HASHER(fixed0_127_t);
    DECLARE_HASHER(fixed1_126_t);
    DECLARE_HASHER(fixed63_64_t);
    DECLARE_HASHER(fixed64_63_t);
    DECLARE_HASHER(fixed127_0_t);
    
    DECLARE_HASHER(ufixed0_128_t);
    DECLARE_HASHER(ufixed1_127_t);
    DECLARE_HASHER(ufixed64_64_t);
    DECLARE_HASHER(ufixed128_0_t);
    #endif
    
    #undef DECLARE_HASHER
    
} /* namespace std */

namespace {
    extern "C" {
        int main() {
            
            using u8fixed8u_t = sg14::fixed_repr_t<std::uint16_t, -8>;
            
            #if defined(_SG14_FIXED_POINT_128)
            std::cout << "Size of __int128_t: " << sizeof(__int128_t) << std::endl;
            #endif
            
            std::cout << "Size of sg14::ufixed8_8_t: " << sizeof(sg14::ufixed8_8_t) << std::endl;
            std::cout << "Size of sg14::fixed15_16_t: " << sizeof(sg14::fixed15_16_t) << std::endl;
            std::cout << "Size of sg14::ufixed16_16_t: " << sizeof(sg14::ufixed16_16_t) << std::endl;
            #if defined(_SG14_FIXED_POINT_128)
            std::cout << "Size of sg14::ufixed64_64_t: " << sizeof(sg14::ufixed64_64_t) << std::endl;
            #endif
            std::cout << "Size of u8fixed8u_t: " << sizeof(u8fixed8u_t) << std::endl;
            
            std::cout << "Alignment of sg14::ufixed8_8_t: " << alignof(sg14::ufixed8_8_t) << std::endl;
            std::cout << "Alignment of sg14::fixed15_16_t: " << alignof(sg14::fixed15_16_t) << std::endl;
            std::cout << "Alignment of sg14::ufixed16_16_t: " << alignof(sg14::ufixed16_16_t) << std::endl;
            #if defined(_SG14_FIXED_POINT_128)
            std::cout << "Alignment of sg14::ufixed64_64_t: " << alignof(sg14::ufixed64_64_t) << std::endl;
            #endif
            std::cout << "Alignment of u8fixed8u_t: " << alignof(u8fixed8u_t) << std::endl;
            
            using u8fixed8_t = sg14::ufixed8_8_t;
            // using s15fixed16_t = sg14::fixed15_16_t;
            // using u16fixed16_t = sg14::ufixed16_16_t;
            
            u8fixed8_t fixedeight_one = u8fixed8_t(3.141);
            u8fixed8_t fixedeight_two = u8fixed8_t::from_data(111);
            u8fixed8u_t fixedeight_union_one;
            u8fixed8u_t fixedeight_union_two;
            
            std::cout << "Fixed point numbers: " << fixedeight_one
                                         << ", " << fixedeight_two
                                                 << std::endl;
            
            
            fixedeight_union_one.value = u8fixed8u_t::fixed_t(3.141);
            fixedeight_union_two.repr = 111;
            
            std::cout << "Fixed point union 1: " << fixedeight_union_one
                                         << " (" << fixedeight_union_one.repr << ")"
                                                 << std::endl;
            
            std::cout << "Fixed point union 2: " << fixedeight_union_two
                                         << " (" << fixedeight_union_two.repr << ")"
                                                 << std::endl;
            
            return 0;
            
        }
    }
} /* namespace (anon.) */

#endif  // defined(_SG14_FIXED_POINT)

