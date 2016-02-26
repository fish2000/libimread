
#ifndef LIBIMREAD_EXT_JSON_JSON11_H_
#define LIBIMREAD_EXT_JSON_JSON11_H_

/// This file is part of json11 project (https://github.com/borisgontar/json11).
/// 
/// Copyright (c) 2013 Boris Gontar.
/// This library is free software; you can redistribute it and/or modify
/// it under the terms of the MIT license. See LICENSE for details.
/// Version 0.6.5, 2013-11-07

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <regex>
#include <cfloat>
#include <cstdint>
#include <utility>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>
#include <libimread/ext/filesystem/path.h>

using Base = std::integral_constant<uint8_t, 1>;
using BaseType = Base::value_type;

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
        /// schema is forward-declared:
        struct Schema;
        struct Node {
            static constexpr Type typecode = Type::JSNULL;
            unsigned refcnt;
            Node(unsigned init = 0);
            virtual ~Node();
            virtual Type type() const { return Type::JSNULL; }
            virtual void print(std::ostream& out) const { out << "null"; }
            virtual void traverse(void (*f)(const Node*)) const { f(this); }
            virtual bool contains(const Node* that) const { return false; }
            virtual bool operator==(const Node& that) const {
                return this == &that;
            }
            virtual bool is_schema() const { return false; }
            void unref();
            const char* typestr() const { return tc::typestr(this->type()); }
            virtual void validate(const Schema& schema,
                                  std::vector<const Node*>&) const;
            static Node null, undefined;
        };
        
        struct Bool : Node {
            static constexpr Type typecode = Type::BOOLEAN;
            Bool(bool x) { refcnt = 1; }
            Type type() const override { return Type::BOOLEAN; }
            void print(std::ostream &out) const override;
            static Bool T;
            static Bool F;
        };
        
        struct Number : Node {
            static constexpr Type typecode = Type::NUMBER;
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
            Type type() const override { return Type::NUMBER; }
            void print(std::ostream& out) const override;
            bool operator==(const Node& that) const override;
            void validate(const Schema& schema,
                          std::vector<const Node*>&) const override;
        };
        
        struct String : Node {
            static constexpr Type typecode = Type::STRING;
            std::string value;
            String(std::string const& s)
                :value(s)
                {}
            String(char const* cs)
                :value(cs)
                {}
            String(std::istream&);
            Type type() const override { return Type::STRING; }
            void print(std::ostream& out) const override;
            bool operator==(const Node& that) const override;
            void validate(const Schema& schema,
                          std::vector<const Node*>&) const override;
        };
        
        struct Array : Node {
            static constexpr Type typecode = Type::ARRAY;
            std::vector<Node*> list;
            virtual ~Array();
            Type type() const override { return Type::ARRAY; }
            void print(std::ostream&) const override;
            void traverse(void (*f)(const Node*)) const override;
            void add(Node*);
            void ins(int, Node*);
            void del(int);
            void repl(int, Node*);
            bool contains(const Node*) const override;
            bool operator==(const Node& that) const override;
            void validate(const Schema& schema,
                          std::vector<const Node*>&) const override;
        };
        
        struct Object : Node {
            static constexpr Type typecode = Type::OBJECT;
            std::map<const std::string*, Node*> map;
            virtual ~Object();
            Type type() const override { return Type::OBJECT; }
            void print(std::ostream&) const override;
            void traverse(void (*f)(const Node*)) const override;
            Node* get(const std::string&) const;
            void set(const std::string&, Node*);
            bool contains(const Node*) const override;
            bool operator==(const Node& that) const override;
            void validate(const Schema& schema,
                          std::vector<const Node*>&) const override;
        };
        
        struct Schema : Node {
            static constexpr Type typecode = Type::SCHEMA;
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
            std::regex* pattern = nullptr; // regex
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
            Type type() const override { return Type::SCHEMA; }
            bool is_schema() const override { return true; }
        };
        
        class Property {
            Node* host;
            std::string key;
            int index;
            Json target() const;
            
            public:
                Property(Node*, const std::string&);
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
                Property operator[](const std::string& k) { return target()[k]; }
                Property operator[](const char* k) { return (*this)[std::string(k)]; }
                Property operator[](int i) { return target()[i]; }
                Json operator=(const Json&);
                Json operator=(const Property&);
                bool operator==(const Json& js) const { return (Json)(*this) == js; }
                bool operator!=(const Json& js) const { return !(*this == js); }
                std::vector<std::string> keys() { return target().keys(); }
                bool has(const std::string& key) const { return target().has(key); }
                bool has(const char* key) const { return target().has(std::string(key)); }
            
            friend std::ostream &operator<<(std::ostream& out, const Property& p) {
                return out << (Json)p;
            }
            
            friend Json;
        };
        
        static std::set<std::string> keyset; /// all propery names
        static int level;                    /// for pretty printing
        
        Json(Node* node) {
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
        
        Node* root;
        
    public:
        /// constructors
        Json() { (root = &Node::null)->refcnt++; }
        Json(const Json& that);
        Json(Json&& that) noexcept;
        Json(std::istream&, bool full=true); // parse
        virtual ~Json();
        
        /// assignment
        Json &operator=(const Json&);
        Json &operator=(Json&&) noexcept;
        
        /// more constructors
        Json(int x)                 { (root = new Number(x))->refcnt++; }
        Json(float x)               { (root = new Number(x))->refcnt++; }
        Json(const std::string& s)  { (root = new String(s))->refcnt++; }
        Json(bool x)                { (root = (x ? &Bool::T : &Bool::F))->refcnt++; }
        Json(long x)                { (root = new Number(x))->refcnt++; }
        Json(long long x)           { (root = new Number(x))->refcnt++; }
        Json(double x)              { (root = new Number(x))->refcnt++; }
        Json(long double x)         { (root = new Number(x))->refcnt++; }
        Json(const char* s)         { (root = new String(s))->refcnt++; }
        Json(const Property& p)     { (root = p.target().root)->refcnt++; }
        Json(std::initializer_list<Json>);
        
        explicit Json(uint8_t x)    { (root = new Number(x))->refcnt++; }
        explicit Json(uint16_t x)   { (root = new Number(x))->refcnt++; }
        explicit Json(int8_t x)     { (root = new Number(x))->refcnt++; }
        explicit Json(int16_t x)    { (root = new Number(x))->refcnt++; }
        
        /// dynamic type info
        Array* mkarray();
        Object* mkobject();
        Type type() const                   { return root->type(); }
        const char* typestr() const         { return root->typestr(); }
        std::string typestring() const      { return std::string(root->typestr()); }
        
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
        Json& set(std::string key, const Json& val);
        Json  get(const std::string& key) const;
        bool  has(const std::string& key) const;
        Json& set(const char* key, const Json& val) { return set(std::string(key), val); }
        Json  get(const char* key) const            { return get(std::string(key)); }
        bool  has(const char* key) const            { return has(std::string(key)); }
        Json  update(const Json& other) const;
        std::vector<std::string> keys();
        
        /// traverse
        using JSONNode = Node;
        void traverse(void (*f)(const JSONNode*)) {
            root->traverse(f);
        }
        
        /// cast operations
        template <typename T> inline
        decltype(auto) cast(const std::string& key) const {
            using rT = std::remove_reference_t<T>;
            return static_cast<rT>(get(key));
        }
        
        template <typename T> inline
        decltype(auto) cast(const std::string& key,
                            T default_value) const {
            using rT = std::remove_reference_t<T>;
            if (!has(key)) { return static_cast<rT>(default_value); }
            return cast<T>(key);
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(const std::string& key) const {
            using rT = tc::idx::helper_t<t>;
            return static_cast<rT>(get(key));
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(const std::string& key,
                                 tc::idx::helper_t<t> default_value) const {
            if (!has(key)) { return default_value; }
            return json_cast<t>(key);
        }
        
        /// array operations
        Json& operator<<(const Json&);
        Json&  insert(int index, const Json&);
        Json&   erase(int index);
        Json& replace(int index, const Json&);
        Json  extend(Json const&) const;
        Json& append(Json const&);
        int    index(Json const&) const;
        Json     pop();
        
        /// subscripting
        std::size_t size() const;
        Json::Property operator[](const std::string&);
        Json::Property operator[](const char* k) { return (*this)[std::string(k)]; }
        Json::Property operator[](int);
        
        /// stringification
        std::string stringify() const { return format(); }
        std::string format() const;
        friend std::ostream &operator<<(std::ostream&, const Json&);
        friend std::istream &operator>>(std::istream&, Json&);
        
        /// hashing
        std::size_t hash(std::size_t H = 0) const {
            hash::rehash<std::string>(H, format());
            return H;
        }
        
        /// boolean comparison
        bool operator==(const Json&) const;
        bool operator!=(const Json& that) const { return !(*this == that); }
    
        /// schema hooks
        bool to_schema(std::string* reason);
        bool valid(Json& schema, std::string* reason = nullptr);
        
        /// input parsing
        static Json null, undefined;
        static Json parse(const std::string&);
        static Json parse(const char* json) { return parse(std::string(json)); }
        
        static Json array() { return new Array(); }   // returns empty array
        static Json object() { return new Object(); } // returns empty object
        static int indent;                            // for pretty printing
        
        struct parse_error : im::JSONParseError {
            unsigned line = 0, col = 0;
            parse_error(const char* msg, std::istream& in);
            parse_error(const std::string &msg, std::istream& in);
        };
        
        struct use_error : im::JSONUseError {
            use_error(const char* msg)
                :im::JSONUseError(msg)
                {}
            use_error(const std::string& msg)
                :im::JSONUseError(msg)
                {}
        };
        
        using JSONSchema = Schema;
};

// using Schema = Json::JSONSchema;

template <> inline
decltype(auto) Json::cast<filesystem::path>(const std::string& key) const {
    return filesystem::path(static_cast<std::string>(get(key)));
}
template <> inline
decltype(auto) Json::cast<const char*>(const std::string& key) const {
    return static_cast<std::string>(get(key)).c_str();
}
template <> inline
decltype(auto) Json::cast<char*>(const std::string& key) const {
    return const_cast<char*>(static_cast<std::string>(get(key)).c_str());
}

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
