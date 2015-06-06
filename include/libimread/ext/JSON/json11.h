
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

using Base = std::integral_constant<uint8_t, 1>;
using BaseType = Base::value_type;

class Json; /// FORWARD!!!

enum Type : BaseType { JSNULL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT };

namespace tc {
    
    inline const char* typestr(Type t) {
        switch (t) {
            case Type::JSNULL:          return "NULL";
            case Type::BOOLEAN:         return "BOOLEAN";
            case Type::NUMBER:          return "NUMBER";
            case Type::STRING:          return "STRING";
            case Type::ARRAY:           return "ARRAY";
            case Type::OBJECT:          return "OBJECT";
        }
        imread_raise_default(JSONOutOfRange);
    }
    
    namespace idx {
        template <Type t>
        struct helper;
        
        template <>
        struct helper<Type::JSNULL> {
            typedef Json value_type;
        };
        template <>
        struct helper<Type::BOOLEAN> {
            typedef bool value_type;
        };
        template <>
        struct helper<Type::NUMBER> {
            typedef long double value_type;
        };
        template <>
        struct helper<Type::STRING> {
            typedef std::string value_type;
        };
        template <>
        struct helper<Type::ARRAY> {
            typedef Json value_type;
        };
        template <>
        struct helper<Type::OBJECT> {
            typedef Json value_type;
        };
        
        template <Type t>
        using helper_t = typename helper<t>::value_type;
        using c_str_t = decltype(std::declval<std::string>().c_str());
    }
    
    template <typename T>
    struct is_json_type {
        typedef std::decay_t<T> decayed_t;
        struct helper_t : std::is_same<decayed_t, Base::value_type> {};
        typedef typename helper_t::value_type value_type;
        static constexpr value_type value = helper_t::value;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
    };
    
    template <typename S>
    struct is_c_str {
        typedef std::remove_pointer_t<std::decay_t<S>> underlying_t;
        typedef std::remove_pointer_t<std::decay_t<idx::c_str_t>> underlying_return_t;
        struct helper_t : std::is_same<underlying_t, underlying_return_t> {};
        typedef typename helper_t::value_type value_type;
        static constexpr value_type value = helper_t::value;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
    };
    
    template <typename F>
    struct caster {
        
        private:
            F&& func;
            
        public:
            explicit caster(F&& f)
                :func(std::forward<F>(f))
                {}
            
            /// Most everything (but NOT C-style strings)
            template <typename T, typename dT = std::decay_t<T>>
            typename std::enable_if_t<!is_c_str<T>::value, dT>
            operator()(const std::string &key) const {
                return static_cast<dT>(std::forward<F>(func)(key));
            }
            
            /// C strings (const char *, char[], etcetc)
            template <typename T>
            typename std::enable_if_t<is_c_str<T>::value, idx::c_str_t>
            operator()(const std::string &key) const {
                return static_cast<std::string>(std::forward<F>(func)(key)).c_str();
            }
            
            /// JSON Type enum values (see above)
            template <Type t = Type::STRING>
            auto operator[](const std::string &key) const ->
            decltype(idx::helper_t<t>()) {
                return static_cast<idx::helper_t<t>>(std::forward<F>(func)(key));
            }
        
        private:
            caster(const caster&);
            caster(caster&&);
            caster &operator=(const caster&);
            caster &operator=(caster&&);
            
    };
    
}


class Json {
    private:
        struct Schema; // forward dcl
        struct Node {
            unsigned refcnt;
            Node(unsigned init = 0);
            virtual ~Node();
            virtual Type type() const { return Type::JSNULL; }
            virtual void print(std::ostream &out) const { out << "null"; }
            virtual void traverse(void (*f)(const Node *)) const { f(this); }
            virtual bool contains(const Node *that) const { return false; }
            virtual bool operator==(const Node &that) const {
                return this == &that;
            }
            virtual bool is_schema() const { return false; }
            void unref();
            const char *typestr() const { return tc::typestr(this->type()); }
            virtual void validate(const Schema &schema,
                                  std::vector<const Node *> &) const;
            static Node null, undefined;
        };
        
        struct Bool : Node {
            Bool(bool x) { refcnt = 1; }
            Type type() const override { return Type::BOOLEAN; }
            void print(std::ostream &out) const override;
            static Bool T;
            static Bool F;
        };
        
        struct Number : Node {
            long double value;
            int prec;
            Number(int x) {
                prec = -1;
                value = x;
            }
            Number(float x) {
                prec = FLT_DIG;
                value = x;
            }
            explicit Number(long double x) {
                prec = LDBL_DIG;
                value = x;
            }
            explicit Number(double x) {
                prec = DBL_DIG;
                value = x;
            }
            explicit Number(long long x) {
                prec = DBL_DIG;
                value = x;
            }
            explicit Number(long x) {
                prec = -1;
                value = x;
            }
            explicit Number(uint8_t x) {
                prec = -1;
                value = static_cast<int>(x);
            }
            explicit Number(uint16_t x) {
                prec = -1;
                value = static_cast<int>(x);
            }
            explicit Number(int8_t x) {
                prec = -1;
                value = static_cast<int>(x);
            }
            explicit Number(int16_t x) {
                prec = -1;
                value = static_cast<int>(x);
            }
            Number(std::istream &);
            Type type() const override { return Type::NUMBER; }
            void print(std::ostream &out) const override;
            bool operator==(const Node &that) const override;
            void validate(const Schema &schema,
                          std::vector<const Node *> &) const override;
        };
        
        struct String : Node {
            std::string value;
            String(std::string s) { value = s; }
            String(const char *cs) { value = std::string(cs); }
            String(std::istream &);
            Type type() const override { return Type::STRING; }
            void print(std::ostream &out) const override;
            bool operator==(const Node &that) const override;
            void validate(const Schema &schema,
                          std::vector<const Node *> &) const override;
        };
        
        struct Array : Node {
            std::vector<Node *> list;
            virtual ~Array();
            Type type() const override { return Type::ARRAY; }
            void print(std::ostream &) const override;
            void traverse(void (*f)(const Node *)) const override;
            void add(Node *);
            void ins(int, Node *);
            void del(int);
            void repl(int, Node *);
            bool contains(const Node *) const override;
            bool operator==(const Node &that) const override;
            void validate(const Schema &schema,
                          std::vector<const Node *> &) const override;
        };
        
        struct Object : Node {
            std::map<const std::string *, Node *> map;
            virtual ~Object();
            Type type() const override { return Type::OBJECT; }
            void print(std::ostream &) const override;
            void traverse(void (*f)(const Node *)) const override;
            Node *get(const std::string &) const;
            void set(const std::string &, Node *);
            bool contains(const Node *) const override;
            bool operator==(const Node &that) const override;
            void validate(const Schema &schema,
                          std::vector<const Node *> &) const override;
        };
        
        struct Schema : Node {
            Schema(Node *);
            virtual ~Schema();
            std::string uri;
            std::string s_type;
            Array *s_enum = nullptr;
            std::vector<Schema *> allof;
            std::vector<Schema *> anyof;
            std::vector<Schema *> oneof;
            Schema *s_not = nullptr;
            long double max_num = LDBL_MAX;
            long double min_num = -LDBL_MAX;
            long double mult_of = 0;
            bool max_exc = false, min_exc = false;
            unsigned long max_len = UINT32_MAX;
            unsigned long min_len = 0;
            std::regex *pattern = nullptr; // regex
            Schema *item = nullptr;
            std::vector<Schema *> items;
            Schema *add_items = nullptr;
            bool add_items_bool = false;
            bool unique_items = false;
            Object *props = nullptr;
            Object *pat_props = nullptr;
            Schema *add_props = nullptr;
            bool add_props_bool = false;
            Array *required = nullptr;
            Object *deps = nullptr;
            Object *defs = nullptr;
            Node *deflt = nullptr;
            bool is_schema() const { return true; }
        };
        
        class Property {
            Node *host;
            std::string key;
            int index;
            Json target() const;
            
            public:
                Property(Node *, const std::string &);
                Property(Node *, int);
                explicit Property(Node *n, const char *c)
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
                // explicit operator uint8_t()     { return target(); }
                // explicit operator uint16_t()    { return target(); }
                // explicit operator int8_t()      { return target(); }
                // explicit operator int16_t()     { return target(); }
                Property operator[](const std::string &k) { return target()[k]; }
                Property operator[](const char *k) { return (*this)[std::string(k)]; }
                Property operator[](int i) { return target()[i]; }
                Json operator=(const Json &);
                Json operator=(const Property &);
                bool operator==(const Json &js) const { return (Json)(*this) == js; }
                bool operator!=(const Json &js) const { return !(*this == js); }
                std::vector<std::string> keys() { return target().keys(); }
                bool has(const std::string &key) const { return target().has(key); }
                bool has(const char *key) const { return target().has(std::string(key)); }
            
            friend std::ostream &operator<<(std::ostream &out, const Property &p) {
                return out << (Json)p;
            }
            
            friend Json;
        };
        Array *mkarray();
        Object *mkobject();
        static std::set<std::string> keyset; // all propery names
        static int level;                    // for pretty printing
        
        Json(Node *node) {
            (root = (node == nullptr ? &Node::null : node))->refcnt++;
        }
        Node *root;
        
    public:
        // constructors
        Json() { (root = &Node::null)->refcnt++; }
        Json(const Json &that);
        Json(Json &&that);
        Json(std::istream &, bool full = true); // parse
        virtual ~Json();
        
        // initializers
        Json &operator=(const Json &);
        Json &operator=(Json &&);
        
        // more constructors
        Json(int x)                 { (root = new Number(x))->refcnt++; }
        Json(float x)               { (root = new Number(x))->refcnt++; }
        Json(const std::string &s)  { (root = new String(s))->refcnt++; }
        Json(bool x)                { (root = (x ? &Bool::T : &Bool::F))->refcnt++; }
        Json(long x)                { (root = new Number(x))->refcnt++; }
        Json(long long x)           { (root = new Number(x))->refcnt++; }
        Json(double x)              { (root = new Number(x))->refcnt++; }
        Json(long double x)         { (root = new Number(x))->refcnt++; }
        Json(const char *s)         { (root = new String(s))->refcnt++; }
        Json(const Property &p)     { (root = p.target().root)->refcnt++; }
        Json(std::initializer_list<Json>);
        
        explicit Json(uint8_t x)    { (root = new Number(x))->refcnt++; }
        explicit Json(uint16_t x)   { (root = new Number(x))->refcnt++; }
        explicit Json(int8_t x)     { (root = new Number(x))->refcnt++; }
        explicit Json(int16_t x)    { (root = new Number(x))->refcnt++; }
        
        // casts
        Type type() const                   { return root->type(); }
        const char* typestr() const         { return root->typestr(); }
        std::string typestring() const      { return std::string(root->typestr()); }
        
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
        
        // object
        Json &set(std::string key, const Json &val);
        Json get(const std::string &key) const;
        bool has(const std::string &key) const;
        Json &set(const char *key, const Json &val) { return set(std::string(key), val); }
        Json get(const char *key) const             { return get(std::string(key)); }
        bool has(const char *key) const             { return has(std::string(key)); }
        std::vector<std::string> keys();
        
        template <typename T> inline
        decltype(auto) cast(const std::string &key) const {
            return static_cast<std::remove_reference_t<T>>(get(key));
        }
        
        template <typename T> inline
        decltype(auto) cast(const std::string &key, T default_value) const {
            if (!has(key)) { return static_cast<std::remove_reference_t<T>>(default_value); }
            return cast<T>(key);
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(const std::string &key) const {
            return static_cast<tc::idx::helper_t<t>>(get(key));
        }
        
        template <Type t = Type::STRING> inline
        decltype(auto) json_cast(const std::string &key, tc::idx::helper_t<t> default_value) const {
            if (!has(key)) { return default_value; }
            return json_cast<t>(key);
        }
        
        // array
        Json &operator<<(const Json &);
        void insert(int index, const Json &);
        void erase(int index);
        Json &replace(int index, const Json &);
        // subscript
        std::size_t size() const;
        Json::Property operator[](const std::string &);
        Json::Property operator[](const char *k) { return (*this)[std::string(k)]; }
        Json::Property operator[](int);
        // stringify
        std::string stringify() { return format(); }
        std::string format();
        friend std::ostream &operator<<(std::ostream &, const Json &);
        friend std::istream &operator>>(std::istream &, Json &);
        // compare
        bool operator==(const Json &) const;
        bool operator!=(const Json &that) const { return !(*this == that); }
    
        // schema
        bool to_schema(std::string *reason);
        bool valid(Json &schema, std::string *reason = nullptr);
        
        static Json null, undefined;
        static Json parse(const std::string &);
        static Json parse(const char *json) { return parse(std::string(json)); }
        static Json array() { return new Array(); }   // returns empty array
        static Json object() { return new Object(); } // returns empty object
        static int indent;                            // for pretty printing
        
        struct parse_error : im::JSONParseError {
            unsigned line = 0, col = 0;
            parse_error(const char *msg, std::istream &in);
            parse_error(const std::string &msg, std::istream &in);
        };
        
        struct use_error : im::JSONUseError {
            use_error(const char *msg)
                :im::JSONUseError(msg)
                {}
            use_error(const std::string &msg)
                :im::JSONUseError(msg)
                {}
        };
};

template <> inline
decltype(auto) Json::cast<const char*>(const std::string &key) const {
    return static_cast<std::string>(get(key)).c_str();
}
template <> inline
decltype(auto) Json::cast<char*>(const std::string &key) const {
    return const_cast<char*>(static_cast<std::string>(get(key)).c_str());
}

#endif /// LIBIMREAD_EXT_JSON_JSON11_H_
