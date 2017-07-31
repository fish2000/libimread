/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_INCLUDE_OPTIONS_HH_
#define IMREAD_INCLUDE_OPTIONS_HH_

#include <string>
#include <regex>
#include <tuple>
#include <sstream>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <initializer_list>

#include <libimread/ext/JSON/json11.h>
#include <libimread/store.hh>

using detail::stringvec_t;
using stringmap_t = store::stringmapper::stringmap_t;
using prefixset_t  = std::unordered_set<std::string>;
using prefixgram_t = std::unordered_map<std::string, std::size_t>;
using stringpair_t = std::pair<std::string, std::string>;
using string_init_t = std::initializer_list<std::string>;
using stringpair_init_t = std::initializer_list<stringpair_t>;
using prefixpair_t = std::pair<prefixset_t, stringvec_t>;
using patternmap_t = std::unordered_map<std::string, std::regex>;
using ratios_t = std::tuple<double, double, int, int, int>;

namespace im {
    
    namespace detail {
        using store::detail::kDefaultSep;
        static constexpr std::string::value_type kDefaultSepChar = kDefaultSep[0];
    }
    
    struct Options;
    
    struct OptionsList final : public Json {
        
        public:
            using Json::null;
            using Json::undefined;
            using Json::root;
            using Json::hash;
            using Json::operator[];
        
        public:
            OptionsList();
            
        public:
            OptionsList(stringvec_t const&);
            OptionsList(stringvec_t&&) noexcept;
            
        public:
            OptionsList(Json const& other);
            OptionsList(Json&& other) noexcept;
            explicit OptionsList(Options const&);
            explicit OptionsList(Options&&) noexcept;
            OptionsList(OptionsList const&);
            OptionsList(OptionsList&&) noexcept;
            OptionsList(std::istream& is, bool full = true);
            OptionsList(string_init_t);
            OptionsList(stringpair_init_t);
            virtual ~OptionsList();
        
        public:
            void swap(OptionsList&) noexcept;
            friend void swap(OptionsList&, OptionsList&) noexcept;
            OptionsList& operator=(string_init_t);
            OptionsList& operator=(stringpair_init_t);
            OptionsList& operator=(Json const&);
            OptionsList& operator=(Json&&) noexcept;
            OptionsList& operator=(OptionsList const&);
            OptionsList& operator=(OptionsList&&) noexcept;
        
        public:
            virtual bool can_store() const noexcept;
            
        public:
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            OptionsList(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
        
        public:
            static OptionsList parse(std::string const&);
            
        public:
            std::size_t count() const;
    };
    
    namespace detail {
        using listpair_t = std::pair<OptionsList, OptionsList>;
    }
    
    struct Options final : public Json, public store::stringmapper {
        
        public:
            using Json::null;
            using Json::undefined;
            using Json::root;
            using Json::hash;
            
        public:
            using store::stringmapper::empty;
            using store::stringmapper::clear;
            using store::stringmapper::cache;
        
        public:
            /// Here, we must exclude our template untyped-method
            /// declaration macro, because one of the template methods
            /// it declares is a method named “update” – which
            /// conflicts with the existing non-virtual non-template
            /// base class method `Json::update(…)`.
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(Options);
            DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(Options);
        
        public:
            Options();
            
        protected:
            /// These hidden constructors are delegates, designed to
            /// receive a reference to the cache of the object from which
            /// we’re copying or moving. I don’t like having that specific
            /// implementation detail hanging out for everyone to see,
            /// and/or abuse, so they’re not available for public access.
            Options(stringmap_t const&);
            Options(stringmap_t&&) noexcept;
            
        public:
            Options(Json const&);
            Options(Json&&) noexcept;
            Options(Options const&);
            Options(Options&&) noexcept;
            Options(std::istream& is, bool full = true);
            Options(stringpair_init_t);
            Options(detail::listpair_t);
            explicit Options(OptionsList&&, OptionsList&&);
            virtual ~Options();
        
        public:
            void swap(Options&) noexcept;
            friend void swap(Options&, Options&) noexcept;
            Options& operator=(stringpair_init_t);
            Options& operator=(Json const&);
            Options& operator=(Json&&) noexcept;
            Options& operator=(Options const&);
            Options& operator=(Options&&) noexcept;
        
        public:
            virtual bool can_store() const noexcept;
        
        public:
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            Options(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
        
        public:
            static Options parse(std::string const&);
        
        public:
            /// The core store::stringmapper API, implemented in terms
            /// of a bunch of eerily similar Json base class methods:
                          virtual bool has(std::string const&) const;
            virtual std::string const& get(std::string const&) const;
                  virtual std::string& get(std::string const&);
                          virtual bool set(std::string const&,
                                           std::string const&);
                          virtual bool set(std::string const&,
                                           Json const&);
                          virtual bool del(std::string const&);
                   virtual std::size_t count() const;
                   virtual stringvec_t list() const;
        
        public:
                   virtual std::size_t count(std::regex const& pattern) const;                              /// #keys matching a given pattern
                  virtual std::size_t count(std::string const& prefix,                                      /// #keys with a given prefix
                                            std::string const& separator) const;
            virtual std::size_t prefixcount(std::string const& prefix,
                                            std::string const& separator = detail::kDefaultSep) const;      /// #keys with a given prefix
             virtual std::size_t prefixcount(std::string::value_type sep = detail::kDefaultSepChar) const;  /// #keys SANS prefix (per separator)
        
        public:
             prefixpair_t prefixset(std::string const& separator = detail::kDefaultSep) const;
            prefixgram_t prefixgram(std::string const& separator = detail::kDefaultSep) const;
                    ratios_t ratios(std::string const& separator = detail::kDefaultSep) const;
        
        public:
            Options subset(std::regex const& pattern, bool defix = true) const;
            Options subset(std::string const& prefix, bool defix = true,
                                    std::string const& separator = detail::kDefaultSep) const;
            OptionsList keylist() const;
            OptionsList valuelist() const;
            detail::listpair_t items() const;
        
    };
    
     std::string get_optional_string(Options const& opts, std::string const& key);
    const char* get_optional_cstring(Options const& opts, std::string const& key);
                int get_optional_int(Options const& opts, std::string const& key,   int const default_value = 0);
              bool get_optional_bool(Options const& opts, std::string const& key,  bool const default_value = false);
    
} /* namespace im */

namespace std {
    
    /// std::hash specialization for im::OptionsList and im::Options
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::OptionsList> {
        
        typedef im::OptionsList argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& optlist) const;
    };
    
    template <>
    struct hash<im::Options> {
        
        typedef im::Options argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& opts) const;
    };
    
} /* namespace std */

#endif /// IMREAD_INCLUDE_OPTIONS_HH_