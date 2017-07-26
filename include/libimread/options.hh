/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH_
#define IMREAD_OPTIONS_HH_

#include <string>
#include <sstream>
#include <libimread/ext/JSON/json11.h>
#include <libimread/store.hh>

using detail::stringvec_t;

namespace im {
    
    struct Options final : public Json, public store::stringmapper {
        
        public:
            using Json::null;
            using Json::undefined;
        
        public:
            /// Here, we must exclude our template untyped-method
            /// declaration macro, because the a method named “update”
            /// is one of the template methods it declares – which
            /// conflicts with the existing non-virtual non-template
            /// base class method `Json::update(…)`.
            DECLARE_STRINGMAPPER_TEMPLATE_CONSTRUCTORS(Options);
            DECLARE_STRINGMAPPER_TEMPLATE_TYPED_METHODS(Options);
        
        public:
            Options();
            Options(Json const&);
            Options(Json&&) noexcept;
            Options(std::istream& is, bool full = true);
            virtual ~Options();
            
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
    };
    
    struct OptionsList final : public Json {
        
        public:
            using Json::null;
            using Json::undefined;
        
        public:
            OptionsList();
            OptionsList(Json const& other);
            OptionsList(Json&& other) noexcept;
            OptionsList(std::istream& is, bool full = true);
            virtual ~OptionsList();
            
        public:
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            OptionsList(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
        
        public:
            static OptionsList parse(std::string const&);
    };
    
    std::string get_optional_string(Options const& opts,  std::string const& key);
    const char* get_optional_cstring(Options const& opts, std::string const& key);
            int get_optional_int(Options const& opts,     std::string const& key,   int const default_value = 0);
           bool get_optional_bool(Options const& opts,    std::string const& key,  bool const default_value = false);
    
}

#endif /// IMREAD_OPTIONS_HH_