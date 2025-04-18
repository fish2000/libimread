
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
#include <map>
#include <set>
#include <string>
#include <ostream>
#include <stdfloat>
#include <exception>
#include <functional>
#include <type_traits>
#include <initializer_list>

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

enum Type : BaseType { JSNULL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT, POINTER, SCHEMA };
enum NodeType : BaseType { ROOT, LEAF, STEM };

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
        
        template <> struct helper<Type::JSNULL>  { using value_type = std::nullptr_t; };
        template <> struct helper<Type::BOOLEAN> { using value_type = bool; };
        template <> struct helper<Type::NUMBER>  { using value_type = long double; };
        template <> struct helper<Type::STRING>  { using value_type = std::string; };
        template <> struct helper<Type::ARRAY>   { using value_type = std::vector<Json>; };
        template <> struct helper<Type::OBJECT>  { using value_type = std::map<std::string, Json>; };
        template <> struct helper<Type::POINTER> { using value_type = std::add_pointer_t<void>; };
        template <> struct helper<Type::SCHEMA>  { using value_type = Schema; };
        
        template <Type t>
        using helper_t = typename helper<t>::value_type;
    }
    
}

namespace im {
    struct Options;
    struct OptionsList;
}

namespace store {
    class stringmapper;
}

class Json {
    
    friend struct im::Options;
    friend struct im::OptionsList;
    friend class store::stringmapper;
    
    protected:
        /// Schema is forward-declared:
        struct Schema;
        friend struct Schema;
        
        /// Node is forward-declared too,
        /// for the function-pointer signatures that follow:
        struct Node;
        friend struct Node;
        using nodevec_t  = std::vector<Node*>;
        using nodecvec_t = std::vector<Node const*>;
        using nodemap_t  = std::map<std::string const*, Node*>;
        using nodepair_t = std::pair<std::string const*, Node*>;
        
        /// Property is also forward-declared, and friendly:
        class Property;
        friend class Property;
    
    public:
        using jsonlist_t = std::initializer_list<Json>;
        using jsonvec_t  = std::vector<Json>;
        using jsonmap_t  = std::map<std::string, Json>;
        using jsonpair_t = std::pair<std::string, Json>;
        
        /// Originally this was declared as: `void (*f)(Node const*)` ...
        using traverser_t       = std::function<void(Node const*, Type, NodeType)>;
        using named_traverser_t = std::function<void(Node const*, char const*)>;
        
    protected:
        struct Node {
            static constexpr Type typecode = Type::JSNULL;
            Node(std::size_t init = 0);
            virtual ~Node();
            virtual Type type() const;
            virtual NodeType nodetype() const;
            virtual void print(std::ostream&) const;
            virtual void traverse(traverser_t, bool is_root = false) const;
            virtual void traverse(named_traverser_t, char const* name = nullptr) const;
            virtual bool contains(Node const*) const;
            virtual bool operator==(Node const&) const;
            virtual void validate(Schema const&, nodecvec_t&) const;
            virtual bool is_schema() const;
            mutable std::size_t refcnt;
            void unref();
			std::size_t incref();
			std::size_t decref();
            char const* typestr() const;
            static Node null, undefined;
        };
        
    protected:
        struct Bool : public Node {
            static constexpr Type typecode = Type::BOOLEAN;
            Type type() const override;
            Bool(bool x) { refcnt = 1; }
            void print(std::ostream&) const override;
            static Bool T;
            static Bool F;
        };
        
    protected:
        struct Number : public Node {
            static constexpr Type typecode = Type::NUMBER;
            Type type() const override;
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
            explicit Number(unsigned long x)
                :value(static_cast<int>(x))
                {}
            explicit Number(unsigned int x)
                :value(static_cast<int>(x))
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
            bool is_integer() const;
            void print(std::ostream&) const override;
            bool operator==(Node const&) const override;
            void validate(Schema const&, nodecvec_t&) const override;
        };
        
    protected:
        struct String : public Node {
            static constexpr Type typecode = Type::STRING;
            Type type() const override;
            std::string value;
            String(std::string const& s)
                :value(s)
                {}
            String(char const* cs)
                :value(cs)
                {}
            String(std::istream&);
            void print(std::ostream&) const override;
            bool operator==(Node const&) const override;
            void validate(Schema const&, nodecvec_t&) const override;
        };
        
    protected:
        struct Pointer : public Node {
            static constexpr Type typecode = Type::POINTER;
            Type type() const override;
            void* value = nullptr;
            bool destroy = false;
            explicit Pointer(void* voidptr)
                :value(voidptr)
                {}
            explicit Pointer(void* voidptr, bool d)
                :value(voidptr)
                ,destroy(d)
                {}
            template <typename Whatever>
            explicit Pointer(Whatever* whatevs)
                :value(reinterpret_cast<void*>(whatevs))
                {}
            template <typename Whatever>
            explicit Pointer(Whatever* whatevs, bool d)
                :value(reinterpret_cast<void*>(whatevs))
                ,destroy(d)
                {}
            Pointer(std::istream&);
            virtual ~Pointer();
            void print(std::ostream&) const override;
            bool  has() { return value != nullptr; }
            void* get() { return value; }
            void  set(void* v) { value = v; }
            template <typename T>
            std::remove_reference_t<T> cast() {
                using rT = std::remove_reference_t<T>;
                return reinterpret_cast<rT*>(value);
            }
            template <typename T>
            std::remove_reference_t<T> cast(T* default_value) {
                using rT = std::remove_reference_t<T>;
                return value ? reinterpret_cast<rT*>(value) :
                               reinterpret_cast<rT*>(default_value);
            }
            bool operator==(Node const&) const override;
            void validate(Schema const&, nodecvec_t&) const override;
        };
        
    protected:
        struct Array : public Node {
            static constexpr Type typecode = Type::ARRAY;
            nodevec_t list;
            Type type() const override;
            NodeType nodetype() const override;
            virtual ~Array();
            void print(std::ostream&) const override;
            void add(Node*);
            void pop();
            void ins(int, Node*);
            void del(int);
            void repl(int, Node*);
            Node* at(int) const;
            void traverse(traverser_t, bool is_root = false) const override;
            void traverse(named_traverser_t, char const* name = nullptr) const override;
            bool contains(Node const*) const override;
            bool operator==(Node const&) const override;
            void validate(Schema const&, nodecvec_t&) const override;
        };
        
    protected:
        struct Object : public Node {
            static constexpr Type typecode = Type::OBJECT;
            nodemap_t map;
            Type type() const override;
            NodeType nodetype() const override;
            virtual ~Object();
            void print(std::ostream&) const override;
            Node* get(std::string const&) const;
            void  set(std::string const&, Node*);
            bool  del(std::string const&);
            Node* pop(std::string const&);
            void traverse(traverser_t, bool is_root = false) const override;
            void traverse(named_traverser_t, char const* name = nullptr) const override;
            bool contains(Node const*) const override;
            bool operator==(Node const&) const override;
            void validate(Schema const&, nodecvec_t&) const override;
        };
        
    protected:
        struct Schema : public Node {
            using schemavec_t = std::vector<Schema*>;
            static constexpr Type typecode = Type::SCHEMA;
            Type type() const override;
            bool is_schema() const override;
            Schema(Node*);
            virtual ~Schema();
            std::string uri;
            std::string s_type;
            Array* s_enum = nullptr;
            schemavec_t allof;
            schemavec_t anyof;
            schemavec_t oneof;
            Schema* s_not = nullptr;
            float64_t max_num = LDBL_MAX;
            float64_t min_num = -LDBL_MAX;
            float64_t mult_of = 0;
            bool max_exc = false;
            bool min_exc = false;
            uint64_t max_len = UINT32_MAX;
            uint64_t min_len = 0;
            void* pattern = nullptr; /// std::regex
            Schema* item = nullptr;
            schemavec_t items;
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
        
    protected:
        class Property {
            
            friend class ::Json;
            
            protected:
                Node* host;
                std::string key;
                int kidx;
                
            public:
                Json target() const;
            
            public:
                Property(Node*, std::string const&);
                Property(Node*, int);
                explicit Property(Node*, const char*);
                
            public:
                // operator Json() const           { return target(); }
                explicit operator int()                  { return static_cast<int>(target()); }
                explicit operator float()                { return static_cast<float>(target()); }
                explicit operator std::string() const    { return static_cast<std::string>(target()); }
                explicit operator bool()        { return static_cast<bool>(target()); }
                explicit operator long()        { return static_cast<long>(target()); }
                explicit operator long long()   { return static_cast<long long>(target()); }
                explicit operator double()      { return static_cast<double>(target()); }
                explicit operator long double() { return static_cast<long double>(target()); }
                
            public:
                Property operator[](std::string const& k) { return target()[k]; }
                Property operator[](const char* k) { return (*this)[std::string(k)]; }
                Property operator[](int i) { return target()[i]; }
                
            public:
                Json operator=(Json const&);
                Json operator=(Property const&);
                
            public:
                bool operator==(Json const& js) const { return (Json)(*this) == js; }
                bool operator!=(Json const& js) const { return !(*this == js); }
                detail::stringvec_t keys() { return target().keys(); }
                bool has(std::string const& key) const { return target().has(key); }
            
            public:
                friend std::ostream& operator<<(std::ostream& out, Property const& p) {
                    return out << (Json)p;
                }
        };
        
    protected:
        static detail::stringset_t keyset;   /// all propery names
        static int level;                    /// for pretty printing
        static int indent;                   /// for pretty printing
        
    protected:
        Json(Node* node) {
            (root = (node == nullptr ? &Node::null : node))->incref();
        }
        
    protected:
        Node* root;
        
    public:
        /// constructors
        Json() { (root = &Node::null)->incref(); }
        Json(Json const&);
        Json(Json&&) noexcept;
        Json(std::istream&, bool full = true); /// parse
        virtual ~Json();
        
        /// assignment
        Json& operator=(Json const&);
        Json& operator=(Json&&) noexcept;
        Json& reset(); /// reassigns `null` to root node
        
    public:
        /// more constructors
        Json(int x)                 { (root = new Number(x))->incref(); }
        Json(float x)               { (root = new Number(x))->incref(); }
        Json(std::string const& s)  { (root = new String(s))->incref(); }
        Json(bool x)                { (root = (x ? &Bool::T : &Bool::F))->incref(); }
        Json(long x)                { (root = new Number(x))->incref(); }
        Json(long long x)           { (root = new Number(x))->incref(); }
        Json(unsigned int x)        { (root = new Number(x))->incref(); }
        Json(unsigned long x)       { (root = new Number(x))->incref(); }
        Json(double x)              { (root = new Number(x))->incref(); }
        Json(long double x)         { (root = new Number(x))->incref(); }
        Json(char const* s)         { (root = new String(s))->incref(); }
        Json(Property const& p)     { (root = p.target().root)->incref(); }
        
        explicit Json(uint8_t x)    { (root = new Number(x))->incref(); }
        explicit Json(uint16_t x)   { (root = new Number(x))->incref(); }
        explicit Json(int8_t x)     { (root = new Number(x))->incref(); }
        explicit Json(int16_t x)    { (root = new Number(x))->incref(); }
        
                 Json(jsonlist_t);
        explicit Json(jsonvec_t const&);
        explicit Json(jsonmap_t const&);
        
        /// implict constructor template -- matches anything with
        /// a `to_json()` member function, which it calls on construction.
        /// ... I stole this one weird trick from the other json11: 
        ///     https://github.com/dropbox/json11/blob/master/json11.hpp#L92-L94
        
    public:
        template <typename ConvertibleType,
                  typename = decltype(&ConvertibleType::to_json)>
        Json(ConvertibleType const& convertible)
            :Json(convertible.to_json())
            {}
        
        /// These next two crafty vector- and mapping-converter constructor
        /// templates are also stolen from the other json11 (as above).
        /// ... after all I steal from the best like always dogg you know it
        
    public:
        template <typename Vector,
                  typename std::enable_if_t<
                           std::is_constructible<Json,        typename Vector::value_type>::value,
                      int> = 0>
        explicit Json(Vector const& vector)
            :Json(jsonvec_t(vector.begin(), vector.end()))
            {}
        
        template <typename Mapping,
                  typename std::enable_if_t<
                           std::is_constructible<std::string, typename Mapping::key_type>::value &&
                           std::is_constructible<Json,        typename Mapping::mapped_type>::value,
                      int> = 0>
        explicit Json(Mapping const& mapping)
            :Json(jsonmap_t(mapping.begin(), mapping.end()))
            {}
        
    public:
        /// dynamic type info
        bool is_integer() const;
        Array* mkarray();
        Array* mkarray() const;
        Object* mkobject();
        Object* mkobject() const;
        Type type() const;
        char const* typestr() const;
        std::string typestring() const;
        
    public:
        /// conversion operators
        explicit operator int() const;
        explicit operator float() const;
        explicit operator std::string() const;
        explicit operator bool() const;
        explicit operator uint8_t() const;
        explicit operator uint16_t() const;
        explicit operator int8_t() const;
        explicit operator int16_t() const;
        explicit operator long() const;
        explicit operator long long() const;
        explicit operator double() const;
        explicit operator long double() const;
        explicit operator void*() const;
        
    public:
        /// dictionary operations (or "object properties" in JS-Ville)
        Json& set(std::string const&, Json const&);
        Json  get(std::string const&) const;
        bool  has(std::string const&) const;
        bool  remove(std::string const&);
        Json  update(Json const&) const;
        Json  pop(std::string const&);
        Json  pop(std::string const&, Json const&);
        
    public:
        detail::stringvec_t keys() const;
        detail::stringvec_t subgroups() const;
        static detail::stringvec_t allkeys(); /// return a copy of the keyset
        Json  values() const;     /// returned object is typed as an array
        
    public:
        /// traversal
        using JSONNode = Node; /// legacy
        void traverse(traverser_t) const;
        void traverse(named_traverser_t) const;
        
    public:
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
        
    public:
        /// array operations
        Json& operator<<(Json const&);
        Json&  insert(int, Json const&);
        Json&   erase(int);
        Json& replace(int, Json const&);
        Json  extend(Json const&) const;
        Json& append(Json const&);
        int    index(Json const&) const;
        Json      at(int) const;
        Json     pop();
        
    public:
        /// subscripting
        std::size_t size() const;
        Json::Property operator[](std::string const&);
        Json::Property operator[](int);
        Json::Property operator[](std::string const&) const;
        Json::Property operator[](int) const;
        
    public:
        /// stringification
        std::string stringify() const;
        std::string format() const;
        friend std::ostream& operator<<(std::ostream&, Json const&);
        friend std::istream& operator>>(std::istream&, Json&);
        
    public:
        /// hashing
        std::size_t hash(std::size_t H = 0) const;
        
    public:
        /// boolean comparison
        bool operator==(Json const&) const;
        bool operator!=(Json const&) const;
    
    public:
        /// schema hooks
        bool to_schema(std::string* reason);
        bool valid(Json& schema, std::string* reason = nullptr);
        
    public:
        /// input parsing
        static Json null, undefined;
        static Json parse(std::string const&);
        static Json array() { return new Array(); }    // returns empty array
        static Json object() { return new Object(); }  // returns empty object
        
    public:
        /// file I/O: dump and load
        Json const& dump(std::string const& dest, bool overwrite = false) const;
        std::string dumptmp() const;
        static Json load(std::string const& source);
        
    public:
        struct parse_error : std::runtime_error {
            unsigned line = 0, col = 0;
            parse_error(std::string const& msg, std::istream& in);
        };
        
    public:
        struct use_error : std::exception {
            use_error(std::string const&)
                :std::exception()
                {}
        };
    
}; /* class Json */

template <>
filesystem::path    Json::cast<filesystem::path>(std::string const& key) const;

template <>
const char*         Json::cast<const char*>(std::string const& key) const;

template <>
char*               Json::cast<char*>(std::string const& key) const;

template <>
std::size_t         Json::cast<std::size_t>(std::string const& key) const;

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
    
} /* namespace std */

#endif /// LIBIMREAD_EXT_JSON_JSON11_H_
