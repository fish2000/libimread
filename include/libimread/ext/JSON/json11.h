
#ifndef LIBIMREAD_EXT_JSON_JSON11_H_
#define LIBIMREAD_EXT_JSON_JSON11_H_

/// This file is part of json11 project (https://github.com/borisgontar/json11).
/// 
/// Copyright (c) 2013 Boris Gontar.
/// This library is free software; you can redistribute it and/or modify
/// it under the terms of the MIT license. See LICENSE for details.
/// Version 0.6.5, 2013-11-07

#include <cstdint>
#include <cfloat>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <type_traits>
#include <initializer_list>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

using Base = std::integral_constant<uint8_t, 1>;
using BaseType = Base::value_type;

namespace filesystem {
    class path;
}

namespace detail {
    using stringvec_t = std::vector<std::string>;
    using stringset_t = std::set<std::string>;
}

/// FORWARD!!!
class Json;
class Schema;

enum Type : BaseType { JSNULL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT, SCHEMA };

namespace tc {
    
    /// Return a string description of the typecode
    char const* typestr(Type t);
    
    namespace idx {
        
        /// namespaced compile-time type mapping,
        /// in support of enum-based cast ops
        /// e.g. node.json_cast<Type::SOMETHING>();
        /// ... the whole idea is basically a cheap knockoff
        /// of the similar scheme in Halide/src/Type.h
        
        template <Type t>
        struct helper;
        
        template <> struct helper<Type::JSNULL>  { using value_type = Json; };
        template <> struct helper<Type::BOOLEAN> { using value_type = bool; };
        template <> struct helper<Type::NUMBER>  { using value_type = long double; };
        template <> struct helper<Type::STRING>  { using value_type = std::string; };
        template <> struct helper<Type::ARRAY>   { using value_type = Json; };
        template <> struct helper<Type::OBJECT>  { using value_type = Json; };
        template <> struct helper<Type::SCHEMA>  { using value_type = Schema; };
        
        template <Type t>
        using helper_t = typename helper<t>::value_type;
    }
    
}


class Json {
    
    private:
        /// Schema is forward-declared:
        struct Schema;
        
        /// Node is forward-declared too,
        /// for the function-pointer signatures that follow:
        struct Node;
        using nodevec_t = std::vector<Node*>;
        using nodecvec_t = std::vector<Node const*>;
        using nodemap_t = std::map<std::string const*, Node*>;
        using nodepair_t = std::pair<std::string const*, Node*>;
    
    public:
        /// Originally this was declared as: `void (*f)(Node const*)` ...
        using traverser_t       = std::add_pointer_t<void(Node const*)>;
        using named_traverser_t = std::add_pointer_t<void(Node const*, char const*)>;
        
    private:
        struct Node {
            static constexpr Type typecode = Type::JSNULL;
            unsigned refcnt;
            Node(unsigned init = 0);
            virtual ~Node();
            virtual Type type() const { return Type::JSNULL; }
            virtual void print(std::ostream& out) const { out << "null"; }
            virtual void traverse(traverser_t traverser) const      { traverser(this); }
            virtual void traverse(named_traverser_t named_traverser,
                                  char const* name = nullptr) const { named_traverser(this, name); }
            virtual bool contains(Node const* that) const   { return this == that; }
            virtual bool operator==(Node const& that) const { return this == &that; }
            virtual void validate(Schema const& schema, nodecvec_t&) const;
            virtual bool is_schema() const { return false; }
            void unref();
            char const* typestr() const { return tc::typestr(this->type()); }
            static Node null, undefined;
        };
        
        struct Bool : Node {
            static constexpr Type typecode = Type::BOOLEAN;
            Type type() const override { return Type::BOOLEAN; }
            Bool(bool x) { refcnt = 1; }
            void print(std::ostream& out) const override;
            static Bool T;
            static Bool F;
        };
        
        struct Number : Node {
            static constexpr Type typecode = Type::NUMBER;
            Type type() const override { return Type::NUMBER; }
            long double value;
            int prec = -1;
            Number(int x)
                :value(x)
                {}
            Number(float x)
                :value(x), prec(FLT_DIG)
                {}
            explicit Number(long double x)
                :value(x), prec(LDBL_DIG)
                {}
            explicit Number(double x)
                :value(x), prec(DBL_DIG)
                {}
            explicit Number(long long x)
                :value(x)
                {}
            explicit Number(long x)
                :value(x)
                {}
            explicit Number(uint8_t x)
                :value(static_cast<int>(x))
                {}
            explicit Number(uint16_t x)
                :value(static_cast<int>(x))
                {}
            explicit Number(int8_t x)
                :value(static_cast<int>(x))
                {}
            explicit Number(int16_t x)
                :value(static_cast<int>(x))
                {}
            Number(std::istream&);
            void print(std::ostream& out) const override;
            bool operator==(Node const& that) const override;
            void validate(Schema const& schema, nodecvec_t&) const override;
        };
        
        struct String : Node {
            static constexpr Type typecode = Type::STRING;
            Type type() const override { return Type::STRING; }
            std::string value;
            String(std::string const& s)
                :value(s)
                {}
            String(char const* cs)
                :value(cs)
                {}
            String(std::istream&);
            void print(std::ostream& out) const override;
            bool operator==(Node const& that) const override;
            void validate(Schema const& schema, nodecvec_t&) const override;
        };
        
        struct Array : Node {
            static constexpr Type typecode = Type::ARRAY;
            Type type() const override { return Type::ARRAY; }
            nodevec_t list;
            virtual ~Array();
            void print(std::ostream&) const override;
            void add(Node*);
            void pop();
            void ins(int, Node*);
            void del(int);
            void repl(int, Node*);
            void traverse(traverser_t traverser) const override;
            void traverse(named_traverser_t traverser,
                          char const* name = nullptr) const override;
            bool contains(Node const*) const override;
            bool operator==(Node const& that) const override;
            void validate(Schema const& schema, nodecvec_t&) const override;
        };
        
        struct Object : Node {
            static constexpr Type typecode = Type::OBJECT;
            Type type() const override { return Type::OBJECT; }
            nodemap_t map;
            virtual ~Object();
            void print(std::ostream&) const override;
            Node* get(std::string const&) const;
            void  set(std::string const&, Node*);
            bool  del(std::string const&);
            Node* pop(std::string const&);
            void traverse(traverser_t traverser) const override;
            void traverse(named_traverser_t traverser,
                          char const* name = nullptr) const override;
            bool contains(Node const*) const override;
            bool operator==(Node const& that) const override;
            void validate(Schema const& schema, nodecvec_t&) const override;
        };
        
        struct Schema : Node {
            static constexpr Type typecode = Type::SCHEMA;
            Type type() const override { return Type::SCHEMA; }
            bool is_schema() const override { return true; }
            Schema(Node*);
            virtual ~Schema();
            std::string uri;
            std::string s_type;
            Array* s_enum = nullptr;
            std::vector<Schema*> allof;
            std::vector<Schema*> anyof;
            std::vector<Schema*> oneof;
            Schema* s_not = nullptr;
            long double max_num = LDBL_MAX;
            long double min_num = -LDBL_MAX;
            long double mult_of = 0;
            bool max_exc = false, min_exc = false;
            unsigned long max_len = UINT32_MAX;
            unsigned long min_len = 0;
            void* pattern = nullptr; /// std::regex
            Schema* item = nullptr;
            std::vector<Schema*> items;
            Schema* add_items = nullptr;
            bool add_items_bool = false;
            bool unique_items = false;
            Object* props = nullptr;
            Object* pat_props = nullptr;
            Schema* add_props = nullptr;
            bool add_props_bool = false;
            Array* required = nullptr;
            Object* deps = nullptr;
            Object* defs = nullptr;
            Node* deflt = nullptr;
        };
        
        class Property {
            Node* host;
            std::string key;
            int kidx;
            Json target() const;
            
            public:
                Property(Node*, std::string const&);
                Property(Node*, int);
                explicit Property(Node* n, const char* c)
                    :Property(n, std::string(c))
                    {}
                operator Json() const           { return target(); }
                operator int()                  { return target(); }
                operator float()                { return target(); }
                operator std::string() const    { return target(); }
                explicit operator bool()        { return static_cast<bool>(target()); }
                explicit operator long()        { return static_cast<long>(target()); }
                explicit operator long long()   { return static_cast<long long>(target()); }
                explicit operator double()      { return static_cast<double>(target()); }
                explicit operator long double() { return static_cast<long double>(target()); }
                Property operator[](std::string const& k) { return target()[k]; }
                Property operator[](const char* k) { return (*this)[std::string(k)]; }
                Property operator[](int i) { return target()[i]; }
                Json operator=(Json const&);
                Json operator=(Property const&);
                bool operator==(Json const& js) const { return (Json)(*this) == js; }
                bool operator!=(Json const& js) const { return !(*this == js); }
                detail::stringvec_t keys() { return target().keys(); }
                bool has(std::string const& key) const { return target().has(key); }
                bool has(char const* key) const { return target().has(std::string(key)); }
            
            friend std::ostream& operator<<(std::ostream& out, Property const& p) {
                return out << (Json)p;
            }
            
            friend Json;
        };
        
        static detail::stringset_t keyset;   /// all propery names
        static int level;                    /// for pretty printing
        static int indent;                   /// for pretty printing
        static std::mutex mute;
        
        Json(Node* node) {
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
        
        Node* root;
        
    public:
        /// constructors
        Json() { (root = &Node::null)->refcnt++; }
        Json(Json const& that);
        Json(Json&& that) noexcept;
        Json(std::istream&, bool full=true); // parse
        virtual ~Json();
        
        /// assignment
        Json &operator=(Json const&);
        Json &operator=(Json&&) noexcept;
        
        /// more constructors
        Json(int x)                 { (root = new Number(x))->refcnt++; }
        Json(float x)               { (root = new Number(x))->refcnt++; }
        Json(std::string const& s)  { (root = new String(s))->refcnt++; }
        Json(bool x)                { (root = (x ? &Bool::T : &Bool::F))->refcnt++; }
        Json(long x)                { (root = new Number(x))->refcnt++; }
        Json(long long x)           { (root = new Number(x))->refcnt++; }
        Json(double x)              { (root = new Number(x))->refcnt++; }
        Json(long double x)         { (root = new Number(x))->refcnt++; }
        Json(char const* s)         { (root = new String(s))->refcnt++; }
        Json(Property const& p)     { (root = p.target().root)->refcnt++; }
        Json(std::initializer_list<Json>);
        
        explicit Json(uint8_t x)    { (root = new Number(x))->refcnt++; }
        explicit Json(uint16_t x)   { (root = new Number(x))->refcnt++; }
        explicit Json(int8_t x)     { (root = new Number(x))->refcnt++; }
        explicit Json(int16_t x)    { (root = new Number(x))->refcnt++; }
        
        /// dynamic type info
        Array* mkarray();
        Array* mkarray() const;
        Object* mkobject();
        Object* mkobject() const;
        Type type() const                   { return root->type(); }
        const char* typestr() const         { return root->typestr(); }
        std::string typestring() const      { return root->typestr(); }
        
        /// conversion operators
        operator int() const;
        operator float() const;
        operator std::string() const;
        explicit operator bool() const;
        explicit operator uint8_t() const;
        explicit operator uint16_t() const;
        explicit operator int8_t() const;
        explicit operator int16_t() const;
        explicit operator long() const;
        explicit operator long long() const;
        explicit operator double() const;
        explicit operator long double() const;
        
        /// dictionary operations (or "object properties" in JS-Ville)
        Json& set(std::string const& key, Json const& value);
        Json  get(std::string const& key) const;
        bool  has(std::string const& key) const;
        bool  remove(std::string const& key);
        Json  update(Json const& other) const;
        Json  pop(std::string const& key);
        Json  pop(std::string const& key, Json const& default_value);
        
        detail::stringvec_t keys() const;
        Json  values() const;     /// returned object is typed as an array
        
        /// traversal
        using JSONNode = Node; /// legacy
        void traverse(traverser_t) const;
        void traverse(named_traverser_t) const;
        
        /// cast operations
        template <typename T> inline
        std::remove_reference_t<T> cast(std::string const& key) const {
            using rT = std::remove_reference_t<T>;
            return static_cast<rT>(get(key));
        }
        
        template <typename T> inline
        std::remove_reference_t<T> cast(std::string const& key,
                            T default_value) const {
            using rT = std::remove_reference_t<T>;
            if (!has(key)) { return static_cast<rT>(default_value); }
            return cast<T>(key);
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(std::string const& key) const {
            using rT = tc::idx::helper_t<t>;
            return static_cast<rT>(get(key));
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(std::string const& key,
                                 tc::idx::helper_t<t> default_value) const {
            if (!has(key)) { return default_value; }
            return json_cast<t>(key);
        }
        
        /// array operations
        Json& operator<<(const Json&);
        Json&  insert(int idx, Json const&);
        Json&   erase(int idx);
        Json& replace(int idx, Json const&);
        Json  extend(Json const&) const;
        Json& append(Json const&);
        int    index(Json const&) const;
        Json     pop();
        
        /// subscripting
        std::size_t size() const;
        Json::Property operator[](std::string const&);
        Json::Property operator[](const char* k) { return (*this)[std::string(k)]; }
        Json::Property operator[](int);
        
        /// stringification
        std::string stringify() const { return format(); }
        std::string format() const;
        friend std::ostream &operator<<(std::ostream&, const Json&);
        friend std::istream &operator>>(std::istream&, Json&);
        
        /// hashing
        std::size_t hash(std::size_t H = 0) const;
        
        /// boolean comparison
        bool operator==(Json const&) const;
        bool operator!=(Json const& that) const { return !(*this == that); }
    
        /// schema hooks
        bool to_schema(std::string* reason);
        bool valid(Json& schema, std::string* reason = nullptr);
        
        /// input parsing
        static Json null, undefined;
        static Json parse(std::string const&);
        
        static Json array() { return new Array(); }   // returns empty array
        static Json object() { return new Object(); } // returns empty object
        
        struct parse_error : im::JSONParseError {
            unsigned line = 0, col = 0;
            parse_error(std::string const& msg, std::istream& in);
        };
        
        struct use_error : im::JSONUseError {
            use_error(std::string const& msg)
                :im::JSONUseError(msg)
                {}
        };
        
};

template <>
filesystem::path Json::cast<filesystem::path>(std::string const& key) const;

template <>
const char* Json::cast<const char*>(std::string const& key) const;

template <>
char* Json::cast<char*>(std::string const& key) const;

namespace std {
    
    /// std::hash specialization for Json
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<Json> {
        
        typedef Json argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& json) const {
            return static_cast<result_type>(json.hash());
        }
        
    };
    
}; /* namespace std */

#endif /// LIBIMREAD_EXT_JSON_JSON11_H_
