/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#ifndef JSON11_H_
#define JSON11_H_

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <regex>
#include <cfloat>
#include <stdexcept>
#include <initializer_list>

class Json {
  public:
    enum Type { JSNULL, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    
    
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
#ifdef WITH_SCHEMA
        virtual void validate(const Schema &schema,
                              std::vector<const Node *> &) const;
#endif
#ifdef TEST
        static std::vector<Node *> nodes;
        static void test();
#endif
        static Node null, undefined;
    };
    //
    struct Bool : Node {
        Bool(bool x) { refcnt = 1; }
        Type type() const override { return Type::BOOL; }
        void print(std::ostream &out) const override;
        static Bool T;
        static Bool F;
    };
    //
    struct Number : Node {
        long double value;
        int prec;
        Number(long double x) {
            prec = LDBL_DIG;
            value = x;
        }
        Number(double x) {
            prec = DBL_DIG;
            value = x;
        }
        Number(float x) {
            prec = FLT_DIG;
            value = x;
        }
        Number(long long x) {
            prec = DBL_DIG;
            value = x;
        }
        Number(long x) {
            prec = -1;
            value = x;
        }
        Number(int x) {
            prec = -1;
            value = x;
        }
        Number(std::istream &);
        Type type() const override { return Type::NUMBER; }
        void print(std::ostream &out) const override;
        bool operator==(const Node &that) const override;
#ifdef WITH_SCHEMA
        void validate(const Schema &schema,
                      std::vector<const Node *> &) const override;
#endif
    };
    //
    struct String : Node {
        std::string value;
        String(std::string s) { value = s; }
        String(std::istream &);
        Type type() const override { return Type::STRING; }
        void print(std::ostream &out) const override;
        bool operator==(const Node &that) const override;
#ifdef WITH_SCHEMA
        void validate(const Schema &schema,
                      std::vector<const Node *> &) const override;
#endif
    };
    //
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
#ifdef WITH_SCHEMA
        void validate(const Schema &schema,
                      std::vector<const Node *> &) const override;
#endif
    };
    //
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
#ifdef WITH_SCHEMA
        void validate(const Schema &schema,
                      std::vector<const Node *> &) const override;
#endif
    };
//
#ifdef WITH_SCHEMA
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
#endif
    //
    class Property {
        Node *host;
        std::string key;
        int index;
        Json target() const;

      public:
        Property(Node *, const std::string &);
        Property(Node *, int);
        operator Json() const { return target(); }
        operator bool() { return target(); }
        operator int() { return target(); }
        operator long() { return target(); }
        operator long long() { return target(); }
        operator float() { return target(); }
        operator double() { return target(); }
        operator long double() { return target(); }
        operator std::string() const { return target(); }
        Property operator[](const std::string &k) { return target()[k]; }
        Property operator[](const char *k) { return (*this)[std::string(k)]; }
        Property operator[](int i) { return target()[i]; }
        Json operator=(const Json &);
        Json operator=(const Property &);
        bool operator==(const Json &js) const { return (Json)(*this) == js; }
        bool operator!=(const Json &js) const { return !(*this == js); }
        std::vector<std::string> keys() { return target().keys(); }
        bool has(const std::string &key) const { return target().has(key); }
        friend std::ostream &operator<<(std::ostream &out, const Property &p) {
            return out << (Json)p;
        }
        friend Json;
    };
    Array *mkarray();
    Object *mkobject();
    static std::set<std::string> keyset; // all propery names
    static int level;                    // for pretty printing
    //
    Json(Node *node) {
        (root = (node == nullptr ? &Node::null : node))->refcnt++;
    }
    Node *root;
    //
  public:
    // constructors
    Json() { (root = &Node::null)->refcnt++; }
    Json(const Json &that);
    Json(Json &&that);
    Json(std::istream &, bool full = true); // parse
    virtual ~Json();
    //
    // initializers
    Json &operator=(const Json &);
    Json &operator=(Json &&);
    //
    // more constructors
    Json(bool x) { (root = (x ? &Bool::T : &Bool::F))->refcnt++; }
    Json(int x) { (root = new Number(x))->refcnt++; }
    Json(long x) { (root = new Number(x))->refcnt++; }
    Json(long long x) { (root = new Number(x))->refcnt++; }
    Json(float x) { (root = new Number(x))->refcnt++; }
    Json(double x) { (root = new Number(x))->refcnt++; }
    Json(long double x) { (root = new Number(x))->refcnt++; }
    Json(const std::string &s) { (root = new String(s))->refcnt++; }
    Json(const char *s) { (root = new String(s))->refcnt++; }
    Json(std::initializer_list<Json>);
    Json(const Property &p) { (root = p.target().root)->refcnt++; }
    //
    // casts
    Type type() const { return root->type(); }
    
    operator bool() const;
    operator int() const;
    operator long() const;
    operator long long() const;
    operator float() const;
    operator double() const;
    operator long double() const;
    operator std::string() const;
    //
    // object
    Json &set(std::string key, const Json &val);
    Json get(const std::string &key) const;
    bool has(const std::string &key) const;
    std::vector<std::string> keys();
    
    template <Type t>
    struct type_helper;
    
    template <>
    struct type_helper<Type::JSNULL> {
        typedef Node type;
        typedef nullptr value_type;
    };
    template <>
    struct type_helper<Type::BOOL> {
        typedef Bool type;
        typedef bool value_type;
    };
    template <>
    struct type_helper<Type::NUMBER> {
        typedef Number type;
        typedef long double value_type;
    };
    template <>
    struct type_helper<Type::STRING> {
        typedef String type;
        typedef std::string value_type;
    };
    template <>
    struct type_helper<Type::ARRAY> {
        typedef Array type;
        typedef Json value_type;
    };
    template <>
    struct type_helper<Type::OBJECT> {
        typedef Object type;
        typedef Json value_type;
    };
    
    template <typename T>
    struct cast {
        using dT = typename std::decay<T>::type;
        dT operator()(const std::string &key) const {
            return static_cast<dT>(get(key));
        }
    };
    template <Type t>
    struct cast {
        using rT = typename type_helper<t>::value_type;
        rT operator()(const std::string &key) const {
            return static_cast<rT>(get(key));
        }
    };
    template <>
    struct cast<const char*> {
        const char *operator()(const std::string &key) const {
            return static_cast<std::string>(get(key)).c_str();
        }
    };
    
    //
    // array
    Json &operator<<(const Json &);
    void insert(int index, const Json &);
    void erase(int index);
    Json &replace(int index, const Json &);
    //
    // subscript
    size_t size() const;
    Json::Property operator[](const std::string &);
    Json::Property operator[](const char *k) { return (*this)[std::string(k)]; }
    Json::Property operator[](int);
    //
    // stringify
    std::string stringify() { return format(); }
    std::string format();
    friend std::ostream &operator<<(std::ostream &, const Json &);
    friend std::istream &operator>>(std::istream &, Json &);
    //
    // compare
    bool operator==(const Json &) const;
    bool operator!=(const Json &that) const { return !(*this == that); }
//
#ifdef WITH_SCHEMA
    // schema
    bool to_schema(std::string *reason);
    bool valid(Json &schema, std::string *reason = nullptr);
#endif
    //
    static Json null, undefined;
    static Json parse(const std::string &);
    static Json array() { return new Array(); }   // returns empty array
    static Json object() { return new Object(); } // returns empty object
    static int indent;                            // for pretty printing
    //
    struct parse_error : std::runtime_error {
        unsigned line = 0, col = 0;
        parse_error(const char *msg, std::istream &in);
    };
    struct use_error : std::logic_error {
        use_error(const char *msg) : std::logic_error(msg) {}
        use_error(const std::string msg) : std::logic_error(msg.c_str()) {}
    };
#ifdef TEST
    static void test() { Node::test(); }
#endif
};

#endif /* JSON11_H_ */
