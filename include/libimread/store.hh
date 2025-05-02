/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_STORE_HH_
#define LIBIMREAD_INCLUDE_STORE_HH_

#include <initializer_list>
#include <unordered_map>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <string>
#include <regex>
#include <vector>

#include <libimread/libimread.hpp>

namespace store {
    
    namespace detail {
        
        constexpr char kDefaultSep[] = ":";
        constexpr char kDefaultRep[] = "";
        
        template <typename ValueType> inline
        auto value_for_null() -> ValueType&;
        
        template <> inline
        auto value_for_null<std::string>() -> std::string& {
            static std::string nk{ NULL_STR };
            return nk;
        }
        
    }
    
    template <typename Key, typename Mapped,
              typename Value = std::pair<std::add_const_t<Key>, Mapped>>
    class base {
        
        public:
            using key_type = Key;
            using mapped_type = Mapped;
            using value_type = Value;
            using size_type = std::size_t;
            using hash_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            
            using reference = std::add_lvalue_reference_t<value_type>;
            using rvalue_reference = std::add_rvalue_reference_t<value_type>;
            using const_reference = std::add_lvalue_reference_t<
                                    std::add_const_t<value_type>>;
            
            using key_reference = std::add_lvalue_reference_t<key_type>;
            using key_const_reference = std::add_lvalue_reference_t<
                                        std::add_const_t<key_type>>;
            using key_rvalue_reference = std::add_rvalue_reference_t<key_type>;
            
            using mapped_reference = std::add_lvalue_reference_t<mapped_type>;
            using mapped_const_reference = std::add_lvalue_reference_t<
                                           std::add_const_t<mapped_type>>;
            using mapped_rvalue_reference = std::add_rvalue_reference_t<mapped_type>;
        
        public:
            virtual ~base() {}
            
            template <typename KeyType = Key>
            auto null_key() const -> KeyType& {
                return detail::value_for_null<KeyType>();
            };
            
            template <typename MappedType = Mapped>
            auto null_value() const -> MappedType& {
                return detail::value_for_null<MappedType>();
            };
            
            virtual bool can_store() const noexcept { return false; }
            
            virtual bool empty() const = 0;
            virtual size_type size() const = 0;
            virtual size_type max_size() const noexcept = 0;
            virtual hash_type hash(hash_type) const = 0;
            
            virtual void clear() = 0;
            virtual bool insert(rvalue_reference) = 0;
            virtual bool emplace(reference) { return false; }
            virtual size_type erase(key_const_reference) = 0;
            
            virtual mapped_reference at(key_const_reference) = 0;
            virtual mapped_const_reference at(key_const_reference) const = 0;
            // virtual mapped_reference operator[](key_const_reference) { return null_value(); }
            // virtual mapped_reference operator[](key_rvalue_reference) { return null_value(); }
            virtual size_type count(key_const_reference) const = 0;
    };
    
    class stringmapper : public base<std::string, std::string> {
        
        public:
            using base_t = base<std::string, std::string>;
            using stringvec_t = std::vector<std::string>;
            using stringmap_t = std::unordered_map<std::string, std::string>;
            
            using stringpair_t = std::pair<std::string, std::string>;
            using string_init_t = std::initializer_list<std::string>;
            using stringpair_init_t = std::initializer_list<stringpair_t>;
            using patternmap_t = std::unordered_map<std::string, std::regex>;
            
            using base_t::null_key;
            using base_t::null_value;
        
        public:
            enum class formatter : int8_t {
                undefined   = -1,
                json        = 0,
                plist       = 1,
                urlparam    = 2,
                ini         = 4,
                yaml        = 8,
                pickle      = 16
            };
            
            static constexpr formatter default_format = formatter::json;
        
        public:
            virtual std::string&       get(std::string const& key) = 0;
            virtual std::string const& get(std::string const& key) const = 0;
            virtual bool set(std::string const& key, std::string const& value) = 0;
            virtual bool del(std::string const& key) = 0;
            virtual std::size_t count() const = 0;
            virtual stringvec_t list() const = 0;
        
        public:
            virtual bool operator+=(stringpair_t const&);
            virtual bool operator+=(stringpair_init_t);
            virtual bool operator-=(stringpair_t const&);
            virtual bool operator-=(stringpair_init_t);
            virtual bool operator-=(string_init_t);
        
        public:
            virtual void with_ini(std::string const&);
            virtual void with_json(std::string const&);
            virtual void with_plist(std::string const&);
            virtual void with_urlparam(std::string const&);
            virtual void with_yaml(std::string const&);
            virtual void warm_cache() const;
            virtual stringmap_t& mapping() const;
            virtual std::string mapping_ini() const;
            virtual std::string mapping_json() const;
            virtual std::string mapping_plist() const;
            virtual std::string mapping_urlparam(bool questionmark = false) const;
            virtual std::string mapping_yaml() const;
            virtual std::string to_string() const;
            virtual bool dump(std::string const& destination,
                                bool overwrite = false,
                              formatter format = stringmapper::default_format) const;
        
        public:
            virtual ~stringmapper();
            
            virtual bool empty() const override;
            virtual std::size_t size() const override;
            virtual std::size_t max_size() const noexcept override;
            virtual std::size_t hash(std::size_t H = 0) const override;
            
            virtual void clear() override;
            virtual bool insert(std::pair<const std::string, std::string>&& item) override;
            virtual std::size_t erase(std::string const& key) override;
            virtual std::string& at(std::string const& key) override;
            virtual std::string const& at(std::string const& key) const override;
            // virtual std::string& operator[](std::string const& key) override;
            // virtual std::string& operator[](std::string&& key) override;
            virtual std::size_t count(std::string const& key) const override;
        
        protected:
            mutable stringmap_t cache;
    
    };
    
    namespace detail {
        
        /// “all_of” bit based on http://stackoverflow.com/q/13562823/298171
        template <bool ...booleans>
        struct static_all_of;
        
        /// derive recursively, if the first argument is true:
        template <bool ...tail>
        struct static_all_of<true, tail...> : static_all_of<tail...>
        {};
        
        /// end recursion: false, if first argument is false:
        template <bool ...tail>
        struct static_all_of<false, tail...> : std::false_type
        {};
        
        /// end recursion: true, if all arguments have been exhausted:
        template <>
        struct static_all_of<> : std::true_type
        {};
        
        template <bool ...booleans>
        using static_all_of_t = typename static_all_of<booleans...>::type;
        
        template <bool ...booleans>
        constexpr bool static_all_of_v = static_all_of<booleans...>::value;
        
        template <typename Type, typename ...Requirements>
        struct are_bases_of : static_all_of_t<std::is_base_of<Type,
                                              std::decay_t<Requirements>>::value...>
        {};
    }
    
    template <typename ...Types>
    struct is_stringmapper : detail::are_bases_of<store::stringmapper, Types...>::type
    {};
    
    template <typename ...Types>
    constexpr bool is_stringmapper_v = is_stringmapper<Types...>::value;
    
    template <typename T, typename U>
    void value_copy(T&& from, U&& to) {
        static_assert(store::is_stringmapper_v<T, U>,
                     "store::value_copy() operands must derive from store::stringmapper");
		if (!std::forward<T>(from).empty()) {
			stringmapper::stringvec_t froms(std::forward<T>(from).list());
            for (std::string const& name : froms) { std::forward<U>(to).set(name,
                                                    std::forward<T>(from).get(name)); }
        }
    }
    
    template <typename T, typename U>
    void prefix_copy(T&& from, U&& to, std::string const& prefix,
                                       std::string const& sep = store::detail::kDefaultSep) {
        static_assert(store::is_stringmapper_v<T, U>,
                     "store::prefix_copy() operands must derive from store::stringmapper");
		if (!std::forward<T>(from).empty()) {
			stringmapper::stringvec_t froms(std::forward<T>(from).list());
            for (std::string const& name : froms) { std::forward<U>(to).set(prefix + sep + name,
                                                    std::forward<T>(from).get(name)); }
        }
    }
    
    template <typename T, typename U>
    void defix_copy(T&& from, U&& to, std::string const& prefix,
                                      std::string const& sep = store::detail::kDefaultSep) {
        static_assert(store::is_stringmapper_v<T, U>,
                     "store::defix_copy() operands must derive from store::stringmapper");
 		if (!std::forward<T>(from).empty()) {
 			stringmapper::stringvec_t froms(std::forward<T>(from).list());
            std::regex defix_re("^" + prefix + sep, std::regex::extended);
            for (std::string const& name : froms) { std::forward<U>(to).set(
                                                    std::regex_replace(name, defix_re, ""),
                                                    std::forward<T>(from).get(name)); }
        }
    }
    
    namespace tag {
        struct tagbase                  {};
        struct prefix : public tagbase  {};
        struct defix  : public tagbase  {};
    }
    
    #define DECLARE_STRINGMAPPER_TEMPLATE_METHODS()                                             \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        X update(T&& from) {                                                                    \
            store::value_copy(std::forward<T>(from), *this);                                    \
        }                                                                                       \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        X prefix(T&& from, std::string const& prefix,                                           \
                           std::string const& sep = store::detail::kDefaultSep) {               \
            store::prefix_copy(std::forward<T>(from), *this, prefix, sep);                      \
        }                                                                                       \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        X defix(T&& from, std::string const& prefix,                                            \
                          std::string const& sep = store::detail::kDefaultSep) {                \
            store::defix_copy(std::forward<T>(from), *this, prefix, sep);                       \
        }
    
    #define DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(__typename__)                           \
                                                                                                \
        template <typename ...Args>                                                             \
        static __typename__ load(std::string const& source, Args&& ...args) {                   \
            __typename__ out(std::forward<Args>(args)...);                                      \
            store::value_copy(store::stringmap::load_map(source), out);                         \
            return out;                                                                         \
        }                                                                                       \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, store::stringmap>>         \
        X interpolate(T&& source) {                                                             \
            stringmapper::stringmap_t const& sourcemap = std::forward<T>(source).mapping();     \
            stringmapper::patternmap_t patterns;                                                \
            std::transform(sourcemap.begin(),     sourcemap.end(),                              \
                           std::inserter(patterns, patterns.end()),                             \
                       [&](auto const& item) {                                                  \
                std::regex  re(R"(\$\()" + item.first + R"(\))", std::regex::extended);         \
                std::string text(item.second);                                                  \
                return std::make_pair(std::move(text),                                          \
                                      std::move(re)); });                                       \
            X out;                                                                              \
            for (std::string const& name : std::forward<__typename__>(*this).list()) {          \
                std::string text = std::forward<__typename__>(*this).get(name);					\
                std::for_each(patterns.begin(),                                                 \
                              patterns.end(),                                                   \
                          [&](auto const& item) { if (std::regex_search(text, item.second,      \
                                                      std::regex_constants::match_default)) {   \
                          text = std::regex_replace(text, item.second, item.first); }           \
                });                                                                             \
                std::forward<X>(out).set(name, text);                                           \
            }                                                                                   \
            return out;                                                                         \
        }
    
    #define DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(__typename__)                            \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        explicit __typename__(T&& from)                                                         \
            :__typename__()                                                                     \
            {                                                                                   \
                store::value_copy(std::forward<T>(from), *this);                                \
            }                                                                                   \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        explicit __typename__(T&& from, std::string const& prefix,                              \
                                        std::string const& sep = store::detail::kDefaultSep,    \
                                        store::tag::prefix tag = store::tag::prefix{})          \
            :__typename__()                                                                     \
            {                                                                                   \
                store::prefix_copy(std::forward<T>(from), *this, prefix, sep);                  \
            }                                                                                   \
                                                                                                \
        template <typename T,                                                                   \
                  typename X = typename std::enable_if_t<                                       \
                                        store::is_stringmapper_v<T>, void>>                     \
        explicit __typename__(T&& from, std::string const& prefix,                              \
                                        std::string const& sep,                                 \
                                        store::tag::defix tag)                                  \
            :__typename__()                                                                     \
            {                                                                                   \
                store::defix_copy(std::forward<T>(from), *this, prefix, sep);                   \
            }
    
    #define DECLARE_STRINGMAPPER_TEMPLATES(__typename__)                                        \
                                                                                                \
        DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(__typename__)                                \
        DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(__typename__)                               \
        DECLARE_STRINGMAPPER_TEMPLATE_METHODS()
    
    class xattrmap : public stringmapper {
        
        public:
            /// We only macro-declare xattrmap::update<…>() methods here –
            /// the templated explicit constructors delegate to the default
            /// constructor of the polymorphic union of the base classes,
            /// which OK that’s kind of a thorny assumption to make, yeah,
            /// that whatever bases are underneath a `store::xattrmap` can
            /// just get themselves the fuck constructed without any sort of
            /// questions asked – but so then of course the fucking intermediate
            /// file source/sinks `im::fd_store_sink` and `im::file_source_sink`
            /// furnish no such default constructors, LIKE A COUPLE OF COMMIES.
            /// I mean, erm, that’s why we have two macros, ah. Yes.
            DECLARE_STRINGMAPPER_TEMPLATE_METHODS();
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            /// xattr API
            virtual std::string xattr(std::string const&) const = 0;
            virtual std::string xattr(std::string const&, std::string const&) const = 0;
            virtual int xattrcount() const = 0;
            virtual stringvec_t xattrs() const = 0;
        
        public:
            std::string&       get_force(std::string const&);
            std::string const& get_force(std::string const&) const;
        
        public:
            /// implementation of the stringmapper API, in terms of xattr()/xattrcount()/xattrs()
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
    };
    
    class stringmap final : public stringmapper {
        
        public:
            using stringmapper::cache;
        
        public:
            DECLARE_STRINGMAPPER_TEMPLATES(stringmap);
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            stringmap() noexcept = default;                     /// default (empty) constructor
            stringmap(stringpair_init_t);                       /// construct from initializer list of string pairs
            stringmap(stringmap const&) = default;              /// default copy constructor
            explicit stringmap(std::string const&,              /// construct from serialized string (e.g JSON)
                               stringmapper::formatter format = stringmapper::default_format);
        
        public:
            static stringmap load_map(std::string const&);      /// load from disk-based file and create anew
        
        public:
            virtual void warm_cache() const override;           /// override with inexpensive no-op
            
        public:
            virtual void swap(stringmap&) noexcept;             /// member-noexcept swap
            stringmap& operator=(stringpair_init_t);            /// assign with initializer list of string pairs
            stringmap& operator=(stringmap const&);             /// copy-and-swap copy-assign operator
            stringmap& operator=(stringmap&&) noexcept;         /// cache-map value-exchange move-assign operator
        
        public:
            /// implementation of the stringmapper API, in terms of std::unordered_map<…> API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
    };
    
} /// namespace store

/// operator==(stringmapper&, stringmapper&) and its evil twin,
/// operator!=(stringmapper&, stringmapper&) both short-circut as soon
/// as an inequality is detected; first by checking the key counts, and
/// then while enumerating the two mappers’ values for comparison.

template <typename T,
          typename U,
          typename X = typename std::enable_if_t<
                                store::is_stringmapper_v<T, U>,
                       bool>>
X operator==(T const& lhs,
             U const& rhs) {
    lhs.warm_cache();
    rhs.warm_cache();
    store::stringmapper::stringvec_t keys = lhs.list();
    if ((lhs.count() != rhs.count()) ||
        (lhs.count() != keys.size()) ||
        (rhs.count() != keys.size())) { return false; }
    for (std::string const& key : keys) {
        if (bool(lhs.get(key) != rhs.get(key))) { return false; }
    }
    return true;
}

template <typename T,
          typename U,
          typename X = typename std::enable_if_t<
                                store::is_stringmapper_v<T, U>,
                       bool>>
X operator!=(T const& lhs,
             U const& rhs) {
    lhs.warm_cache();
    rhs.warm_cache();
    store::stringmapper::stringvec_t keys = lhs.list();
    if ((lhs.count() != rhs.count()) ||
        (lhs.count() != keys.size()) ||
        (rhs.count() != keys.size())) { return true; }
    for (std::string const& key : keys) {
        if (bool(lhs.get(key) != rhs.get(key))) { return true; }
    }
    return false;
}

/// operator<(stringmapper&, stringmapper&) is trivially defined,
/// just to get e.g. std::map<stringmapper, …> to work:

template <typename T,
          typename U,
          typename X = typename std::enable_if_t<
                                store::is_stringmapper_v<T, U>,
                       bool>>
X operator<(T const& lhs,
            U const& rhs) { return lhs.count() < rhs.count(); }

/// operator>(stringmapper&, stringmapper&) is also trivially defined --
/// mainly for reciprocality with operator<(…); it is not a meaningful operation:

template <typename T,
          typename U,
          typename X = typename std::enable_if_t<
                                store::is_stringmapper_v<T, U>,
                       bool>>
X operator>(T const& lhs,
            U const& rhs) { return lhs.count() > rhs.count(); }

namespace std {
    
    /// std::hash specialization for store::stringmap
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<store::stringmapper> {
        
        typedef store::stringmapper argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const&) const;
        
    };
    
} /// namespace std

#endif /// LIBIMREAD_INCLUDE_STORE_HH_