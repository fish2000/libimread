// Named tuple for C++
// Example code from http://vitiy.info/
// Written by Victor Laskin (victor.laskin@gmail.com)
// Adapted by Alexander Bohn (fish2000@gmail.com)
// Parts of code were taken from: https://gist.github.com/Manu343726/081512c43814d098fe4b

#include <cstdlib>
#include <utility>

namespace string_id {
        
    namespace detail {
        
        using hash_type = std::uint64_t;
        
        constexpr hash_type fnv_basis = 14695981039346656037ull;
        constexpr hash_type fnv_prime = 109951162821ull;
        
        // FNV-1a 64 bit hash
        constexpr hash_type sid_hash(const char* str, hash_type hash = fnv_basis) noexcept {
            return *str ? sid_hash(str + 1, (hash ^ *str) * fnv_prime) : hash;
        }
        
        // constexpr auto HASH(const char* str) -> std::integral_constant<hash_type, sid_hash(str)> {
        //     return std::integral_constant<hash_type, sid_hash(str)>{};
        // }
        
    } /// namespace detail
    
} /// namespace string_id


namespace namedtup {
    
     /// Named parameter (could be empty!)
    template <typename Hash, typename ...Ts>
    struct named_param : public std::tuple<std::decay_t<Ts>...> {
        
        using hash = Hash;                                                                      ///< key
        
        named_param(Ts&&... ts) : std::tuple<std::decay_t<Ts>...>(std::forward<Ts>(ts)...) {};  ///< constructor
        
        template <typename P>
        named_param<Hash, P> operator=(P&& p) {
            return named_param<Hash, P>(std::forward<P>(p));
        }
        
    }; /// struct named_param<…>
    
    template <typename Hash>
    using make_named_param = named_param<Hash>;
    
    /// Named tuple is just tuple of named params
    template <typename ...Params>
    struct named_tuple : public std::tuple<Params...> {
        
        template <typename ...Args>
        named_tuple(Args&&... args) : std::tuple<Args...>(std::forward<Args>(args)...) {}
        
        static const std::size_t error = -1;
        
        template <std::size_t I = 0, typename Hash>
        constexpr typename std::enable_if<I == sizeof...(Params), const std::size_t>::type
        static get_element_index() {
            return error;
        }
        
        template <std::size_t I = 0, typename Hash>
        constexpr typename std::enable_if<I < sizeof...(Params), const std::size_t>::type
        static get_element_index() {
            using elementType = typename std::tuple_element<I, std::tuple<Params...>>::type;
            //return (typeid(typename elementType::hash) == typeid(Hash)) ? I : get_element_index<I + 1, Hash>();
            return (std::is_same<typename elementType::hash, Hash>::value) ? I : get_element_index<I + 1, Hash>();
        }
        
        template <typename Hash>
        auto const& get() const {
            constexpr std::size_t idx = get_element_index<0, Hash>();
            static_assert((idx != error), "Wrong named tuple key");
            auto& tuple_slot = (std::get<idx>(static_cast<const std::tuple<Params...>&>(*this)));
            return std::get<0>(tuple_slot);
        }
        
        template <typename NP>
        auto const& operator[](NP&& param) {
            return get<typename NP::hash>();
        }
         
    }; /// struct named_tuple
    
    template <typename ...Args>
    auto make_named_tuple(Args&&... args) {
        return namedtup::named_tuple<Args...>(std::forward<Args>(args)...);
    }
    
} /// namespace namedtup

/// “param” is the name of a macro within Halide, soooo:
#define slot(__str__)                                                                       \
                       namedtup::make_named_param<                                          \
                            std::integral_constant<string_id::detail::hash_type,            \
                                                   string_id::detail::sid_hash(__str__)>>{}

// #define SLOT_TYPE(__str__, __type__)                                                        \
//                        namedtup::named_param<                                               \
//                             std::integral_constant<string_id::detail::hash_type,            \
//                                                    string_id::detail::sid_hash(__str__)>,   \
//                             std::remove_reference_t<__type__>>(                             \
//                             std::remove_reference_t<__type__>{})

#define SLOT_TYPE(__str__, __type__) slot(__str__) = std::remove_reference_t<__type__>{}

#define DECLARE_NAMED_TUPLE(...) decltype(namedtup::make_named_tuple(__VA_ARGS__))

// #define param(__str__) namedtup::make_named_param<string_id::detail::HASH(__str__)>{}