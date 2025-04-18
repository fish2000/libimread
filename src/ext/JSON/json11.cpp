/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#include <cmath>
#include <cstdlib>
#include <climits>
#include <sstream>
#include <iomanip>
#include <exception>
#include <stdexcept>
#include <algorithm>

#include <libimread/ext/JSON/json11.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>

Json::Node Json::Node::null(1);
Json::Node Json::Node::undefined(1);
Json Json::null;
Json Json::undefined(&Node::undefined);
Json::Bool Json::Bool::T(true);
Json::Bool Json::Bool::F(false);
detail::stringset_t Json::keyset;
int Json::indent = 4;
int Json::level;

namespace tc {
    
    char const* typestr(Type t) {
        switch (t) {
            case Type::JSNULL:  return "NULL";
            case Type::BOOLEAN: return "BOOLEAN";
            case Type::NUMBER:  return "NUMBER";
            case Type::STRING:  return "STRING";
            case Type::ARRAY:   return "ARRAY";
            case Type::OBJECT:  return "OBJECT";
            case Type::POINTER: return "POINTER";
            case Type::SCHEMA:  return "SCHEMA";
        }
        imread_raise_default(JSONOutOfRange);
    }
    
}

namespace detail {
    using chartraits = std::char_traits<char>;
    
    std::size_t currpos(std::istream& in, std::size_t* pos) {
        std::size_t curr = in.tellg();
        if (pos != nullptr) { *pos = 0; }
        in.seekg(0); // rewind
        if (in.bad()) { return 0; }
        std::size_t count = 0,
                     line = 1,
                      col = 1;
        while (!in.eof() && !in.fail() && ++count < curr) {
            if (in.get() == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        if (pos != nullptr) { *pos = col; }
        return line;
    }
    
    void escape(std::ostream& out, std::string const& str) {
        out << '"';
        for (char c : str) {
            switch (c) {
                case '"':
                    out << '\\' << '"';
                    break;
                case '\\':
                    out << c << c;
                    break;
                case '\'':
                case '\n':
                case '\t':
                case '\b':
                case '\f':
                case '\r':
                case '\v':
                // case '\p':
                // case '\x':
                case '\?':
                    out << '\\' << c;
                    break;
                default:
                    out << c;
                    break;
            }
        }
        out << '"';
    }
    
}

Json::parse_error::parse_error(std::string const& msg, std::istream& in)
    :std::runtime_error(msg)
    {
        line = detail::currpos(in, &col);
    }

/// Node and helper classes
Json::Node::Node(std::size_t init)
    :refcnt(init)
    {}

Json::Node::~Node() {
    imread_assert(this == &null
		       || this == &undefined
			   || this == &Bool::T
			   || this == &Bool::F
			   || refcnt == 0, "Non-static Node has non-zero refcount upon destruction");
}

/// LEGACY FUNCTION:
/// Decrements a nodes’ reference count,
/// deleting the node if the count has hit zero.
/// Returns the current reference count.
void Json::Node::unref() {
    if (this == &null
	 || this == &undefined
	 || this == &Bool::T
	 || this == &Bool::F) { return; }
    imread_assert(refcnt > 0, "Trying to unref() a node whose refcount <= 0");
    if (--refcnt == 0) { delete this; }
}

/// Increments a nodes’ reference count
/// Returns the current reference count.
std::size_t Json::Node::incref() {
    if (this == &null
	 || this == &undefined
	 || this == &Bool::T
	 || this == &Bool::F) { return 1; }
	return ++refcnt;
}

/// Decrements a nodes’ reference count,
/// deleting the node if the count has hit zero.
/// Returns the current reference count.
std::size_t Json::Node::decref() {
    if (this == &null
     || this == &undefined
	 || this == &Bool::T
	 || this == &Bool::F) { return 1; }
    imread_assert(refcnt > 0, "Trying to decref() a node whose refcount <= 0");
    if (--refcnt == 0) { delete this; }
	return refcnt;
}


bool Json::Array::operator==(Node const& that) const {
    if (this == &that) { return true; }
    if (that.type() != Type::ARRAY) { return false; }
    Json::nodevec_t& that_list = static_cast<Array*>(
                                 const_cast<Node*>(&that))->list;
    return std::equal(list.begin(), list.end(),
                 that_list.begin(),
                   [](Node* p, Node* q) { return *p == *q; });
}

bool Json::Object::operator==(Node const& that) const {
    using kv_t = Json::nodepair_t;
    if (this == &that) { return true; }
    if (that.type() != Type::OBJECT) { return false; }
    Json::nodemap_t& that_map = static_cast<Object*>(
                                const_cast<Node*>(&that))->map;
    return std::equal(map.begin(), map.end(),
                 that_map.begin(),
                   [](kv_t const& p,
                      kv_t const& q) { return *p.first == *q.first &&
                                             *p.second == *q.second; });
}

bool Json::Number::operator==(Node const& that) const {
    if (this == &that) { return true; }
    if (that.type() != Type::NUMBER) { return false; }
    Number& numb = *(Number*)&that;
    if (std::fabs(value) < LDBL_EPSILON) {
        return std::fabs(numb.value) < LDBL_EPSILON;
    }
    long double delta = std::fabs((value - numb.value) / value);
    int digs = std::max(prec, numb.prec);
    return delta < std::pow(10, -digs);
}

bool Json::Number::is_integer() const {
    return std::fabs(value - std::floor(value)) <= LDBL_EPSILON;
}

bool Json::is_integer() const {
    if (root->type() != Type::NUMBER) {
        imread_raise(JSONUseError,
            "Json::is_integer() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::NUMBER)");
    }
    return static_cast<Number*>(root)->is_integer();
}

bool Json::Array::contains(Node const* that) const {
    if (that == nullptr) { return false; }
    if (this == that) { return true; }
    for (Node* it : list) {
        if (it->contains(that)) {
            return true;
        }
    }
    return false;
}

bool Json::Object::contains(Node const* that) const {
    if (that == nullptr) { return false; }
    if (this == that) { return true; }
    for (auto const& it : map) {
        if (it.second->contains(that)) {
            return true;
        }
    }
    return false;
}

/** Copy constructor. */
Json::Json(Json const& that) {
    (root = that.root)->incref();
}

/** Move constructor. */
Json::Json(Json&& that) noexcept {
    root = std::move(that.root);
    that.root = nullptr;
}

Json::Json(Json::jsonlist_t arglist) {
    (root = new Array())->incref();
    for (auto const& arg : arglist) {
        *this << arg;
    }
}

Json::Json(Json::jsonvec_t const& argvec) {
    (root = new Array())->incref();
    for (auto const& arg : argvec) {
        *this << arg;
    }
}

Json::Json(Json::jsonmap_t const& propmap) {
    (root = new Object())->incref();
    for (auto const& prop : propmap) {
        set(prop.first, prop.second);
    }
}

/** Copy assignment */
Json& Json::operator=(Json const& that) {
    root->decref();
    (root = that.root)->incref();
    return *this;
}

/** Move assignment */
Json& Json::operator=(Json&& that) noexcept {
    root->decref();
    root = std::move(that.root);
    that.root = nullptr;
    return *this;
}

Json::~Json() {
    if (root != nullptr) { root->decref(); }
}

Json& Json::reset() {
    root->decref();
    (root = &Node::null)->incref();
    return *this;
}

Json::Object* Json::mkobject() {
    if (root->type() == Type::JSNULL) {
        root = new Object();
        root->incref();
    }
    if (root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::mkobject() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    return static_cast<Object*>(root);
}
Json::Object* Json::mkobject() const {
    if (root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::mkobject() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    return static_cast<Object*>(root);
}

Json& Json::set(std::string const& key, Json const& value) {
    if (value.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    mkobject()->set(key, value.root);
    return *this;
}

Json::Array* Json::mkarray() {
    if (root->type() == Type::JSNULL) {
        root = new Array();
        root->incref();
    }
    if (root->type() != Type::ARRAY) {
        imread_raise(JSONUseError,
            "Json::mkarray() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::ARRAY)");
    }
    return static_cast<Array*>(root);
}
Json::Array* Json::mkarray() const {
    if (root->type() != Type::ARRAY) {
        imread_raise(JSONUseError,
            "Json::mkarray() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::ARRAY)");
    }
    return static_cast<Array*>(root);
}

Type        Json::type() const { return root->type(); }
char const* Json::typestr() const { return root->typestr(); }
std::string Json::typestring() const { return root->typestr(); }

Json& Json::operator<<(Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    mkarray()->add(that.root);
    return *this;
}

Json& Json::insert(int idx, Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    mkarray()->ins(idx, that.root);
    return *this;
}

Json& Json::replace(int idx, Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    mkarray()->repl(idx, that.root);
    return *this;
}

Json& Json::erase(int idx) {
    mkarray()->del(idx);
    return *this;
}

Json Json::extend(Json const& other) const {
    if (other.root == nullptr) {
        imread_raise(JSONUseError,
            "Json::extend(other): invalid operand",
            "\t(other->root        == nullptr)");
    }
    if (root->type()       != Type::ARRAY ||
        other.root->type() != Type::ARRAY) {
        imread_raise(JSONUseError,
            "Json::get(key) method not applicable",
         FF("\troot->type()        == Type::%s", root->typestr()),
         FF("\tother->root->type() == Type::%s", other.root->typestr()),
            "\t(Requires both of Type::ARRAY)");
    }
    if (other.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    Json out = Json{};
    auto const& a = static_cast<Array*>(root)->list;
    auto const& b = static_cast<Array*>(other.root)->list;
    auto ap = out.mkarray();
    std::for_each(a.begin(), a.end(),
              [&](Node* item) { ap->add(item); });
    std::for_each(b.begin(), b.end(),
              [&](Node* item) { ap->add(item); });
    return out;
}

Json& Json::append(Json const& other) {
    if (other.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    mkarray()->add(other.root);
    return *this;
}

int Json::index(Json const& other) const {
    auto ap = mkarray();
    auto const& arlist = ap->list;
    if (!root->contains(other.root)) { return -1; }
    int idx = 0;
    for (auto it = arlist.begin();
         it != arlist.end();
         ++it) {
			 if (other.root == *it) { return idx; }
             ++idx;
		 }
    return -1;
}

Json Json::at(int idx) const {
    try {
        return Json(mkarray()->at(idx));
    } catch (std::out_of_range const&) {}
    return Json(null);
}

Json Json::pop() {
    auto ap = mkarray();
    auto const& arlist = ap->list;
    if (arlist.empty()) { return null; }
    Json out(arlist.back());
    ap->pop();
    return out;
}

Json Json::Property::operator=(Json const& that) {
    switch (host->type()) {
        case Type::JSNULL:
            (host = &Node::null)->incref();
            break;
        case Type::BOOLEAN:
            host = that.root == &Bool::T ? &Bool::T : &Bool::F;
            break;
        case Type::NUMBER:
            if (that.is_integer()) {
                static_cast<Number*>(host)->value = static_cast<long>(that);
            } else {
                static_cast<Number*>(host)->value = static_cast<long double>(that);
            }
            break;
        case Type::STRING:
            static_cast<String*>(host)->value = static_cast<std::string>(that);
            break;
        case Type::ARRAY:
            static_cast<Array*>(host)->repl(kidx, that.root);
            break;
        case Type::OBJECT:
            static_cast<Object*>(host)->set(key, that.root);
            break;
        case Type::POINTER:
        case Type::SCHEMA:
        default:
            break;
    }
    return target();
}

Json Json::Property::operator=(Property const& that) {
    return (*this = that.target());
}

Json::Property Json::operator[](int idx) {
    return Property(mkarray(), idx);
}

Json::Property Json::operator[](std::string const& key) {
    return Property(mkobject(), key);
}

Json::Property Json::operator[](int idx) const {
    return Property(mkarray(), idx);
}

Json::Property Json::operator[](std::string const& key) const {
    return Property(mkobject(), key);
}

std::size_t Json::size() const {
    switch (root->type()) {
        case Type::ARRAY:
            return static_cast<Array*>(root)->list.size();
        case Type::OBJECT:
            return static_cast<Object*>(root)->map.size();
        case Type::STRING:
            return static_cast<String*>(root)->value.size();
        case Type::JSNULL:
        case Type::BOOLEAN:
        case Type::NUMBER:
        case Type::POINTER:
        case Type::SCHEMA:
            imread_raise(JSONUseError,
                "Json::size() method not applicable",
             FF("root->type() == Type::%s", root->typestr()),
                "(Requires {Type::OBJECT | Type::ARRAY | Type::STRING})");
    }
}

Json Json::get(std::string const& key) const {
    Node* n = mkobject()->get(key);
    return n == nullptr ? undefined : Json(n);
}

Json Json::update(Json const& other) const {
    if (other.root == nullptr) {
        imread_raise(JSONUseError,
            "Json::update(other) invalid operand",
            "\t(other->root        == nullptr)");
    }
    if (root->type()       != Type::OBJECT ||
        other.root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::update(other) method not applicable",
         FF("\troot->type()        == Type::%s", root->typestr()),
         FF("\tother->root->type() == Type::%s", other.root->typestr()),
            "\t(Requires both of Type::OBJECT)");
    }
    if (other.root->contains(root)) {
        imread_raise(JSONUseError, "cyclic dependency");
    }
    Json out = Json{};
    auto const& a = static_cast<Object*>(root)->map;
    auto const& b = static_cast<Object*>(other.root)->map;
    auto op = out.mkobject();
    std::for_each(b.begin(), b.end(),
            [&op](auto const& pair) {
        op->set(*pair.first, pair.second);
    });
    std::for_each(a.begin(), a.end(),
            [&op](auto const& pair) {
        op->set(*pair.first, pair.second);
    });
    return out;
}

bool Json::has(std::string const& key) const {
    return mkobject()->get(key) != nullptr;
}

bool Json::remove(std::string const& key) {
    return mkobject()->del(key);
}

Json Json::pop(std::string const& key) {
    Node* node = mkobject()->pop(key);      /// refcount unaffected (>= 1)
    if (!node) {
        imread_raise(JSONUseError,
            "Json::pop(key) called with unknown key",
            "\t(root->pop(key)        == nullptr)");
    }
    Json out(node);                         /// constructor increments refcount (>= 2)
    out.root->decref();                     /// back to reality (>= 1)
    return out;
}

Json Json::pop(std::string const& key, Json const& default_value) {
    Node* node = mkobject()->pop(key);      /// refcount unaffected (>= 1)
    if (!node) { return default_value; }
    Json out(node);                         /// constructor increments refcount (>= 2)
    out.root->decref();                     /// back to reality (>= 1)
    return out;
}

Json::Property::Property(Node* node, std::string const& key)
    :host(node), key(key), kidx(-1)
    {
        if (node->type() != Type::OBJECT) {
            imread_raise(JSONUseError,
                "Json::Property::Property(Node*, std::string) method not applicable",
             FF("\tnode->type() == Type::%s", node->typestr()),
                "\t(Requires Type::OBJECT)");
        }
    }

Json::Property::Property(Node* node, int idx)
    :host(node), key{}, kidx(idx)
    {
        if (node->type() != Type::ARRAY) {
            imread_raise(JSONUseError,
                "Json::Property::Property(Node*, int) method not applicable",
             FF("\tnode->type() == Type::%s", node->typestr()),
                "(Requires Type::ARRAY)");
        }
    }

Json::Property::Property(Node* node, const char* ckey)
    :Json::Property::Property(node, std::string(ckey))
    {}

Json Json::Property::target() const {
    switch (host->type()) {
        case Type::OBJECT: return static_cast<Object*>(host)->get(key);
        case Type::ARRAY:   return static_cast<Array*>(host)->at(kidx);
        default: {
            imread_raise(JSONLogicError,
                "Property::operator Json() conversion-operator logic error:",
             FF("\tConverstion attempt made on Property object of Type::%s", host->typestr()),
                "\tProperty object must be Type::OBJECT or Type::ARRAY");
        }
    }
}

detail::stringvec_t Json::keys() const {
    Object* op = mkobject();
    detail::stringvec_t out;
    out.reserve(op->map.size());
    for (auto const& it : op->map) {
        out.emplace_back(*it.first);
    }
    return out;
}

detail::stringvec_t Json::subgroups() const {
    Object* op = mkobject();
    detail::stringvec_t out;
    out.reserve(op->map.size());
    for (auto const& it : op->map) {
        if (it.second->type() == Type::OBJECT &&
            it.second->nodetype() == NodeType::STEM) {
            out.emplace_back(*it.first);
        }
    }
    out.shrink_to_fit();
    return out;
}

detail::stringvec_t Json::allkeys() {
    detail::stringvec_t out;
    out.reserve(keyset.size());
    for (auto const& key : keyset) {
        out.emplace_back(key);
    }
    return out;
}

Json Json::values() const {
    Json out{};
    Array* ap = out.mkarray();
    Object* op = mkobject();
    for (auto const& it : op->map) {
        ap->add(it.second);
    }
    return out;
}

Type Json::Node::type() const {
    return Type::JSNULL;
}

NodeType Json::Node::nodetype() const {
    return NodeType::LEAF;
}

Type Json::Bool::type() const {
    return Type::BOOLEAN;
}

Type Json::Number::type() const {
    return Type::NUMBER;
}

Type Json::String::type() const {
    return Type::STRING;
}

Type Json::Pointer::type() const {
    return Type::POINTER;
}

Type Json::Schema::type() const {
    return Type::SCHEMA;
}

void Json::Node::print(std::ostream& out) const {
    out << "null";
}

void Json::Node::traverse(traverser_t traverser, bool is_root) const {
    traverser(this, this->type(), is_root ? NodeType::ROOT : this->nodetype());
}

void Json::Node::traverse(named_traverser_t named_traverser, char const* name) const {
    named_traverser(this, name);
}

bool Json::Node::contains(Node const* that) const {
    return this == that;
}

bool Json::Node::operator==(Node const& that) const {
    return this == &that;
}

bool Json::Node::is_schema() const {
    return false;
}

bool Json::Schema::is_schema() const {
    return true;
}

char const* Json::Node::typestr() const {
    return tc::typestr(this->type());
}

bool Json::String::operator==(Node const& that) const {
    return this == &that || (
            that.type() == Type::STRING &&
            value == static_cast<String*>(
                     const_cast<Node*>(&that))->value);
}

bool Json::Pointer::operator==(Node const& that) const {
    return this == &that || (
            that.type() == Type::POINTER &&
            &value == &static_cast<Pointer*>(
                       const_cast<Node*>(&that))->value);
}

void Json::Bool::print(std::ostream& out) const {
    out << (this == &Bool::T ? "true" : "false");
}

void Json::Number::print(std::ostream& out) const {
    if (prec > -1) {
        out << std::setprecision(prec);
    }
    out << value;
}

void Json::String::print(std::ostream& out) const {
    detail::escape(out, value);
}

void Json::Pointer::print(std::ostream& out) const {
	out << "#" << reinterpret_cast<std::size_t>(value) << "";
    // out << "#" << (unsigned long)value << "";
}

Type Json::Object::type() const {
    return Type::OBJECT;
}

NodeType Json::Object::nodetype() const {
    return map.empty() ? NodeType::LEAF : NodeType::STEM;
}

void Json::Object::traverse(Json::traverser_t traverser, bool is_root) const {
    traverser(this, this->type(), is_root ? NodeType::ROOT : this->nodetype());
    for (auto it : map) { it.second->traverse(traverser); }
}

void Json::Object::traverse(Json::named_traverser_t named_traverser,
                            char const* name) const {
    named_traverser(this, name);
    for (auto const& it : map) {
        it.second->traverse(named_traverser,
        it.first->c_str());
    }
}

Type Json::Array::type() const {
    return Type::ARRAY;
}

NodeType Json::Array::nodetype() const {
    return list.empty() ? NodeType::LEAF : NodeType::STEM;
}

void Json::Array::traverse(Json::traverser_t traverser, bool is_root) const {
    traverser(this, this->type(), is_root ? NodeType::ROOT : this->nodetype());
    for (auto it : list) { it->traverse(traverser); }
}

void Json::Array::traverse(Json::named_traverser_t named_traverser,
                           char const* name) const {
    std::size_t idx = 0;
    named_traverser(this, name);
    for (auto it : list) {
        std::string nametag = std::string(name == nullptr ? "<nullptr>" : name) + ":" +
                              std::to_string(idx);
        it->traverse(named_traverser,
                     nametag.c_str());
        ++idx;
    }
}

void Json::traverse(Json::traverser_t traverser) const {
    root->traverse(traverser, true);
}

void Json::traverse(Json::named_traverser_t named_traverser) const {
    root->traverse(named_traverser, "root");
}

Json::Pointer::~Pointer() {
    if (destroy) {
        std::free(value);
    }
    value = nullptr;
}

void Json::Object::print(std::ostream& out) const {
    out << '{';
    ++level;
    bool comma = false;
    for (auto const& it : map) {
        if (comma)  { out << ','; }
        if (indent) { out << std::endl
                          << std::string(indent * level, ' '); }
        detail::escape(out, *it.first);
        out << ':';
        if (indent) { out << ' '; }
        it.second->print(out);
        comma = true;
    }
    --level;
    if (indent) { out << std::endl
                      << std::string(indent * level, ' '); }
    out << '}';
}

void Json::Array::print(std::ostream& out) const {
    out << '[';
    ++level;
    bool comma = false;
    for (Node const* it : list) {
        if (comma)  { out << ','; }
        if (indent) { out << std::endl
                          << std::string(indent * level, ' '); }
        it->print(out);
        comma = true;
    }
    --level;
    if (indent) { out << std::endl
                      << std::string(indent * level, ' '); }
    out << ']';
}

Json::Object::~Object() {
    for (auto const& it : map) {
        // Node* np = it.second;
        it.second->decref();
    }
    map.clear();
}

Json::Node* Json::Object::get(std::string const& key) const {
    auto kp = keyset.find(key);
    if (kp == keyset.end()) { return nullptr; }
    auto it = map.find(&*kp);
    if (it == map.end()) { return nullptr; }
    return it->second;
}

void Json::Object::set(std::string const& k, Node* v) {
    imread_assert(v != nullptr, "Json::Object::set(k, v) called with v == nullptr");
    auto kit = keyset.insert(keyset.begin(), k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        // Node* np = it->second;
		it->second.decref()
        it->second = v;
    } else {
        map[&*kit] = v;
    }
    v->incref();
}

bool Json::Object::del(std::string const& k) {
    /// Decrefs Node* for key, erases it from the map and keyset
    auto kit = keyset.find(k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        // Node* np = it->second;
        it->second->decref();
        map.erase(&*kit);
        // keyset.erase(kit);
        return true;
    }
    return false;
}

Json::Node* Json::Object::pop(std::string const& k) {
    /// Returns Node* for key without affecting its refcount
    auto kit = keyset.find(k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        Node* np = it->second;
        map.erase(&*kit);
        // keyset.erase(kit);
        return np;
    }
    return nullptr;
}

Json::Array::~Array() {
    for (Node* it : list) { it->decref(); }
    list.clear();
}

void Json::Array::add(Node* v) {
    imread_assert(v != nullptr, "Json::Array::add(v) called with v == nullptr");
    list.push_back(v);
    v->incref();
}

void Json::Array::pop() {
    Node* v = static_cast<Node*>(list.back());
    v->decref();
    list.pop_back();
}

/** Inserts given Node* before index. */
void Json::Array::ins(int idx, Node* v) {
    imread_assert(v != nullptr, "Json::Array::ins(idx, v) called with v == nullptr");
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    list.insert(list.begin() + idx, v);
    v->incref();
}

void Json::Array::del(int idx) {
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    Node* v = list.at(idx);
    v->decref();
    list.erase(list.begin() + idx);
}

void Json::Array::repl(int idx, Node* v) {
    imread_assert(v != nullptr,
                  "Json::Array::ins(idx, v) called with v == nullptr");
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    Node* u = list.at(idx);
    u->decref();
    list[idx] = v;
    v->incref();
}

Json::Node* Json::Array::at(int idx) const {
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    return list.at(idx);
}

std::ostream& operator<<(std::ostream& out, Json const& json) {
    json.root->print(out);
    return out;
}

std::istream& operator>>(std::istream& in, Json& json) {
    Json t(in);
    json = std::move(t);
    return in;
}

Json::String::String(std::istream& in) {
    int quote = in.get();
    while (!in.eof()) {
        int c = in.get();
        if (c == detail::chartraits::eof()) {
            throw parse_error("unterminated std::string", in);
        }
        if (c == quote) { return; }
        if (c == '\\') {
            c = in.get();
            if (c == quote || c == '\\' || c == '/') { value.push_back(c); }
            else if (c == 'n') { value.push_back('\n'); }
            else if (c == 't') { value.push_back('\t'); }
            else if (c == 'r') { value.push_back('\r'); }
            else if (c == 'b') { value.push_back('\b'); }
            else if (c == 'f') { value.push_back('\f'); }
            else if (c == 'u') {
                unsigned w = 0;
                for (int i = 0; i < 4; i++) {
                    if (!std::isxdigit(c = std::toupper(in.get()))) {
                        throw parse_error("not a hex digit", in);
                    }
                    w = (w << 4) | (std::isdigit(c) ? c - '0' : c - 'A' + 10);
                }
                // garbage in, garbage out
                if (w <= 0x7f) { value.push_back(w); }
                else if (w <= 0x07ff) {
                    value.push_back(0xc0 | ((w >> 6) & 0x1f));
                    value.push_back(0x80 | (w & 0x3f));
                } else {
                    value.push_back(0xe0 | ((w >> 12) & 0x0f));
                    value.push_back(0x80 | ((w >> 6) & 0x3f));
                    value.push_back(0x80 | (w & 0x3f));
                }
            } else {
                throw parse_error("illegal backslash escape", in);
            }
            continue;
        }
        if (std::iscntrl(c)) {
            throw parse_error("control character in std::string", in);
        }
        value.push_back(c);
    }
}

Json::Pointer::Pointer(std::istream& in) {
    char buf[128];
    const char* end = buf+126;
    char* p = buf;
    char c;
    bool leading = true;
    while ((std::isdigit(c = in.get()) || (leading && c == '#')) && p < end) {
        *p++ = c;
        leading = false;
    }
    if ((c == 'x' || c == '0') && p < end) {
        *p++ = c;
        leading = true;
        while ((std::isdigit(c = in.get()) || (leading && (c == '#' || c == '*'))) && p < end) {
            *p++ = c;
            leading = false;
        }
    }
    *p = 0;
    in.putback(c);
    char* eptr = nullptr;
    long double num = std::strtold(buf, &eptr);
    if (eptr != p) {
        throw parse_error("illegal number format", in);
    }
    // value = num;
}

Json::Number::Number(std::istream& in) {
    char buf[128];
    const char* end = buf+126;
    char* p = buf;
    char c;
    bool leading = true;
    while ((std::isdigit(c = in.get()) || (leading && c == '-')) && p < end) {
        *p++ = c;
        leading = false;
    }
    prec = p - buf;
    if (c == '.' && p < end) {
        *p++ = c;
        while (std::isdigit(c = in.get()) && p < end) { *p++ = c; }
        prec = p - buf - 1;
    }
    if ((c == 'e' || c == 'E') && p < end) {
        *p++ = c;
        leading = true;
        while ((std::isdigit(c = in.get()) || (leading && (c == '-' || c == '+'))) && p < end) {
            *p++ = c;
            leading = false;
        }
    }
    *p = 0;
    in.putback(c);
    char* eptr = nullptr;
    long double num = std::strtold(buf, &eptr);
    if (eptr != p) {
        throw parse_error("illegal number format", in);
    }
    value = num;
}

Json::Json(std::istream& in, bool full) {
    char c;
    std::string word;
    root = nullptr;
    if (!(in >> c)) { goto out; }
    if (c == '[') {
        root = new Array();
        root->incref();
        while (in >> c) {
            if (c == ']') { goto out; }
            in.putback(c);
            Json elem(in, false);
            *this << elem;
            in >> c;
            if (c == ',') { continue; }
            in.putback(c);
        }
        throw parse_error("comma or closing bracket expected", in);
    }
    if (c == '{') {
        root = new Object();
        root->incref();
        while (in >> c) {
            if (c == '}') { goto out; }
            in.putback(c);
            Json key(in, false);
            if (key.root->type() != Type::STRING) {
                throw parse_error("std::string expected", in); }
            in >> c;
            if (c != ':') {
                throw parse_error("colon expected", in);
            }
            Json obj(in, false);
            set(static_cast<std::string>(key), obj);
            in >> c;
            if (c == ',') { continue; }
            in.putback(c);
        }
        throw parse_error("comma or closing bracket expected", in);
    }
    if (std::isdigit(c) || c == '-') {
        in.putback(c);
        root = new Number(in);
        root->incref();
        goto out;
    }
    if (c == '\"' || c == '\'') {
        in.putback(c);
        root = new String(in);
        root->incref();
        goto out;
    }
    word.push_back(c);
    for (int i = 0; i < 3; i++) { word.push_back(in.get()); }
    if (word == "null") {
        root = &Node::null;
        root->incref();
        goto out;
    }
    if (word == "true") {
        root = &Bool::T;
        root->incref();
        goto out;
    }
    if (word == "fals" && in.get() == 'e') {
        root = &Bool::F;
        root->incref();
        goto out;
    }
    throw parse_error("json format error", in);
out:
    if (full) {
        if (in.peek() == detail::chartraits::eof()) { return; }
        while (std::isspace(in.get()))
            /* skip */;
        if (in.eof()) { return; }
        throw parse_error("excess text not parsed", in);
    }
}

std::string Json::stringify() const {
    return format();
}

std::string Json::format() const {
    std::ostringstream out;
    out << *this;
    return out.str();
}

Json Json::parse(std::string const& str) {
    std::istringstream is(str);
    Json parsed(is);
    if (is.peek() == detail::chartraits::eof()) { return parsed; }
    while (std::isspace(is.get()))
        /* skip */;
    if (is.eof()) { return parsed; }
    throw parse_error("JSON format error", is);
}

Json const& Json::dump(std::string const& dest, bool overwrite) const {
    using filesystem::path;
    using filesystem::NamedTemporaryFile;
    std::string destination(dest);
    
    try {
        destination = path::expand_user(dest).make_absolute().str();
    } catch (im::FileSystemError& exc) {
        throw;
    }
    
    if (path::exists(destination)) {
        if (!overwrite) {
            imread_raise(JSONIOError,
                "Json::dump(destination, overwrite=false): existant destination",
             FF("\tdest        == %s", dest.c_str()),
             FF("\tdestination == %s", destination.c_str()),
                "\t(Requires overwrite=true or a unique destination)");
        } else if (path::is_directory(destination)) {
            imread_raise(JSONIOError,
                "Json::dump(destination): directory as existant destination",
             FF("\toverwrite   == %s", overwrite ? "true" : "false"),
             FF("\tdest        == %s", dest.c_str()),
             FF("\tdestination == %s", destination.c_str()),
                "\t(Requires overwrite=true with a non-directory destination)");
        }
    }
    
    NamedTemporaryFile tf(".json");
    tf.open();
    tf.stream << *this;
    tf.close();
    
    if (path::exists(destination)) {
        path::remove(destination);
    }
    
    path finalfile = tf.filepath.duplicate(destination);
    if (!finalfile.is_file()) {
        imread_raise(JSONIOError,
            "Json::dump(destination, ...): failed writing to destination",
         FF("\toverwrite   == %s", overwrite ? "true" : "false"),
         FF("\tdest        == %s", dest.c_str()),
         FF("\tdestination == %s", destination.c_str()),
         FF("\tfinalfile   == %s", finalfile.c_str()));
    }
    
    /// return-self by reference
    return *this;
}

std::string Json::dumptmp() const {
    using filesystem::NamedTemporaryFile;
    NamedTemporaryFile tf(".json", false);
    tf.open();
    tf.stream << *this;
    tf.close();
    return std::string(tf.filepath.str());
}

Json Json::load(std::string const& source) {
    using filesystem::path;
    
    if (!path::exists(source)) {
        imread_raise(JSONIOError,
            "Json::load(source): nonexistant source file",
         FF("\tsource == %s", source.c_str()));
    }
    if (!path::is_file_or_link(source)) {
        imread_raise(JSONIOError,
            "Json::load(source): non-file-or-link source file",
         FF("\tsource == %s", source.c_str()));
    }
    if (!path::is_readable(source)) {
        imread_raise(JSONIOError,
            "Json::load(source): unreadable source file",
         FF("\tsource == %s", source.c_str()));
    }
    
    std::fstream stream;
    stream.open(source, std::ios::in);
    if (!stream.is_open()) {
        imread_raise(JSONIOError,
            "Json::load(source): couldn't open a stream to read source file",
         FF("\tsource == %s", source.c_str()));
    }
    
    Json out{};
    stream >> out;
    stream.close();
    
    /// return by value
    return out;
}

std::size_t Json::hash(std::size_t H) const {
    hash::rehash<std::string>(H, format());
    return H;
}

Json::operator std::string() const {
    switch (root->type()) {
        case Type::JSNULL:  return "null";
        case Type::BOOLEAN: return root == &Bool::T ? "true" : "false";
        case Type::NUMBER:  return std::to_string(((Number*)root)->value);
        case Type::STRING:  return ((String*)root)->value;
        case Type::ARRAY:
        case Type::OBJECT:  return format();
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator long double() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0.0 : 1.0;
        case Type::NUMBER:  return is_integer() ? std::round(((Number*)root)->value) : ((Number*)root)->value;
        case Type::STRING:  return std::atof(((String*)root)->value.c_str());
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator double() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0.0 : 1.0;
        case Type::NUMBER:  return is_integer() ? std::round(((Number*)root)->value) : ((Number*)root)->value;
        case Type::STRING:  return std::atof(((String*)root)->value.c_str());
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator float() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0.0 : 1.0;
        case Type::NUMBER:  return is_integer() ? std::round(((Number*)root)->value) : ((Number*)root)->value;
        case Type::STRING:  return std::atof(((String*)root)->value.c_str());
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator int() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return is_integer() ? ((Number*)root)->value : std::lround(((Number*)root)->value);
        case Type::STRING:  return std::stoi(((String*)root)->value);
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator long() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return is_integer() ? ((Number*)root)->value : std::lround(((Number*)root)->value);
        case Type::STRING:  return std::stol(((String*)root)->value);
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator long long() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return is_integer() ? ((Number*)root)->value : std::llround(((Number*)root)->value);
        case Type::STRING:  return std::stoll(((String*)root)->value);
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator uint8_t() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return uint8_t(((Number*)root)->value);
        case Type::STRING:  return uint8_t(std::stoi(((String*)root)->value));
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator uint16_t() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return uint16_t(((Number*)root)->value);
        case Type::STRING:  return uint16_t(std::stoi(((String*)root)->value));
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator int8_t() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return int8_t(((Number*)root)->value);
        case Type::STRING:  return int8_t(std::stoi(((String*)root)->value));
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator int16_t() const {
    switch (root->type()) {
        case Type::JSNULL:  return 0;
        case Type::BOOLEAN: return root == &Bool::T ? 0 : 1;
        case Type::NUMBER:  return int16_t(((Number*)root)->value);
        case Type::STRING:  return int16_t(std::stoi(((String*)root)->value));
        case Type::ARRAY:
        case Type::OBJECT:
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
}

Json::operator void*() const {
    if (root->type() == Type::POINTER) { return ((Pointer*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator bool() const {
    switch (root->type()) {
        case Type::JSNULL:  return false;
        case Type::BOOLEAN: return root == &Bool::T;
        case Type::NUMBER:  return static_cast<bool>(int(((Number*)root)->value));
        case Type::STRING:  return static_cast<bool>(((String*)root)->value.size());
        case Type::ARRAY:   return static_cast<bool>(((Array*)root)->list.size());
        case Type::OBJECT:  return static_cast<bool>(((Object*)root)->map.size());
        case Type::POINTER:
        case Type::SCHEMA:
        default:            imread_raise_default(JSONBadCast);
    }
    imread_raise_default(JSONBadCast);
}

bool Json::operator==(Json const& that) const {
    if (root == that.root) { return true; }
    return *root == *that.root;
}

bool Json::operator!=(Json const& that) const {
    return root !=  that.root &&
        !(*root == *that.root);
}

template <>
filesystem::path    Json::cast<filesystem::path>(std::string const& key) const {
    std::string out = static_cast<std::string>(get(key));
    return filesystem::path(out);
}

template <>
const char*         Json::cast<const char*>(std::string const& key) const {
    std::string out = static_cast<std::string>(get(key));
    return out.c_str();
}

template <>
char*               Json::cast<char*>(std::string const& key) const {
    std::string out = static_cast<std::string>(get(key));
    return const_cast<char*>(out.c_str());
}

template <>
std::size_t         Json::cast<std::size_t>(std::string const& key) const {
    int out = static_cast<int>(get(key));
    return static_cast<std::size_t>(out);
}
