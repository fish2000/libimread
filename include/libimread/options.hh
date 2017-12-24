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

#include <libimread/ext/JSON/json11.h>
#include <libimread/store.hh>

using detail::stringvec_t;                                          /// = std::vector<std::string> (a scommon libimread idiom)
using stringmap_t = store::stringmapper::stringmap_t;               /// = std::unordered_map<std::string, std::string>
using prefixset_t  = std::unordered_set<std::string>;               /// q.v. Options::prefixset(…), Options::prefixgram(…) [ibid.]
using prefixgram_t = std::unordered_map<std::string, std::size_t>;  /// prefixgram_t = “prefix histogram”
using stringpair_t = store::stringmapper::stringpair_t;             /// = std::pair<std::string, std::regex>
using string_init_t = store::stringmapper::string_init_t;           /// = std::initializer_list<std::string>
using stringpair_init_t = store::stringmapper::stringpair_init_t;   /// = std::initializer_list<stringpair_t>
using prefixpair_t = std::pair<prefixset_t, stringvec_t>;           /// convenience, for passing along the vector
using patternmap_t = store::stringmapper::patternmap_t;             /// = std::unordered_map<std::string, std::regex>
using ratios_t = std::tuple<double, double, int, int, int>;         /// q.v. Options::ratios(…) implementation, options.cpp (sub.)

namespace im {
    
    namespace detail {
        using store::detail::kDefaultSep;
        using store::detail::kDefaultRep;
        static constexpr std::string::value_type kDefaultSepChar = kDefaultSep[0];
    }
    
    struct Options;
    
    struct OptionsList final : public Json {
        
        friend struct Options;
        
        public:
            using Json::null;
            using Json::undefined;
            using Json::root;
            using Json::hash;
            using Json::operator[];
        
        public:
            OptionsList();
            
        protected:
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
        
        friend struct OptionsList;
        
        public:
            using Json::null;
            using Json::undefined;
            using Json::root;
            using Json::hash;
            using Json::subgroups;
            
        public:
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
                          virtual bool empty() const;
        
        public:
                   virtual std::size_t count(std::regex const& pattern) const;                              /// #keys matching a given pattern
                  virtual std::size_t count(std::string const& prefix,                                      /// #keys with a given prefix,
                                            std::string const& separator) const;                            ///       and a specific separator
                  virtual std::size_t count(std::string const& key) const;                                  /// #keys with a given prefix
            virtual std::size_t prefixcount(std::string const& prefix,                                      /// #keys with a given prefix,
                                            std::string const& separator = detail::kDefaultSep) const;      ///       optionally specifying separator
             virtual std::size_t prefixcount(std::string::value_type sep = detail::kDefaultSepChar) const;  /// #keys SANS prefix (per separator)
        
        public:
               stringvec_t prefixes(std::string const& separator = detail::kDefaultSep) const;
             prefixpair_t prefixset(std::string const& separator = detail::kDefaultSep) const;
            prefixgram_t prefixgram(std::string const& separator = detail::kDefaultSep) const;
                    ratios_t ratios(std::string const& separator = detail::kDefaultSep) const;
        
        public:
            Options subgroup(std::string const&) const;
            Options subset(std::regex const& pattern, bool defix = true,
                                  std::string const& replacement = detail::kDefaultRep) const;
            Options subset(std::string const& prefix, bool defix = true,
                                    std::string const& separator = detail::kDefaultSep) const;
            Options subset(std::string::value_type sep = detail::kDefaultSepChar) const;
           Options replace(std::regex const& pattern, bool defix = true,
                                  std::string const& replacement = detail::kDefaultRep) const;
            Options underscores_to_dashes() const;
            Options dashes_to_underscores() const;
            
        protected:
            /// Pass-through call to Json::get() -- not for public use:
            Json value(std::string const&) const;
            
        public:
            /// cast operations:
            template <typename T> inline
            std::remove_reference_t<T> cast(std::string const& key) const {
                using rT = std::remove_reference_t<T>;
                return static_cast<rT>(Json::get(key));
            }
            
            template <typename T> inline
            std::remove_reference_t<T> cast(std::string const& key,
                                            T default_value) const {
                using rT = std::remove_reference_t<T>;
                if (!Json::has(key)) { return static_cast<rT>(default_value); }
                return Json::cast<T>(key);
            }
        
        public:
            /// in-place regroup: pull a subgroup down and prefix it:
            Options& regroup(std::string const& subgroupname,
                                       std::string const& prefix,                           /// defaults to the subgroup name
                                    std::string const& separator = detail::kDefaultSep);
            
            /// in-place hoist: pull out a subset and hoist it into a subgroup:
            Options& hoist(std::string const& subsetname,
                                       std::string const& prefix = detail::kDefaultRep,     /// defaults to the subgroup name
                                    std::string const& separator = detail::kDefaultSep);
            
            /// in-place “regroup all”: regroup with current subgroup names,
            /// over and over again, until none are left:
            Options& flatten(std::string const& separator = detail::kDefaultSep);
            
            /// in-place “hoist all”: hoist each current prefix into its own
            /// subgroup, over and over again, until none are left:
            Options& extrude(std::string const& separator = detail::kDefaultSep);
            
        public:
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