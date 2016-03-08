/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#include <cmath>
#include <climits>
#include <sstream>
#include <iomanip>
#include <exception>
#include <stdexcept>
#include <algorithm>

#include <libimread/ext/JSON/json11.h>
#include <libimread/ext/filesystem/path.h>
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
std::mutex Json::mute;

namespace tc {
    
    char const* typestr(Type t) {
        switch (t) {
            case Type::JSNULL:  return "NULL";
            case Type::BOOLEAN: return "BOOLEAN";
            case Type::NUMBER:  return "NUMBER";
            case Type::STRING:  return "STRING";
            case Type::ARRAY:   return "ARRAY";
            case Type::OBJECT:  return "OBJECT";
            case Type::SCHEMA:  return "SCHEMA";
        }
        imread_raise_default(JSONOutOfRange);
    }
    
}

namespace detail {
    using chartraits = std::char_traits<char>;
    
    unsigned currpos(std::istream& in, unsigned* pos) {
        unsigned curr = in.tellg();
        if (pos != nullptr) { *pos = 0; }
        in.seekg(0); // rewind
        if (in.bad()) { return 0; }
        unsigned count = 0,
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
                case '\b':
                    out << '\\' << 'b';
                    break;
                case '\f':
                    out << '\\' << 'f';
                    break;
                case '\n':
                    out << '\\' << 'n';
                    break;
                case '\r':
                    out << '\\' << 'r';
                    break;
                case '\t':
                    out << '\\' << 't';
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
    :im::JSONParseError(msg)
    {
        line = detail::currpos(in, &col);
    }

/// Node and helper classes
Json::Node::Node(unsigned init)
    :refcnt(init)
    {}

Json::Node::~Node() {
    imread_assert(this == &null || this == &undefined || this == &Bool::T || this == &Bool::F || refcnt == 0,
                  "Non-static Node has non-zero refcount upon destruction");
}

void Json::Node::unref() {
    if (this == &null || this == &undefined || this == &Bool::T || this == &Bool::F) {
        return;
    }
    imread_assert(refcnt > 0,
                  "Trying to unref() a node whose refcount <= 0");
    if (--refcnt == 0) { delete this; }
}

bool Json::Array::operator==(Node const& that) const {
    if (this == &that) { return true; }
    if (that.type() != Type::ARRAY) { return false; }
    Json::nodevec_t& that_list = ((Array*)&that)->list;
    return std::equal(list.begin(), list.end(),
                 that_list.begin(),
                   [](Node* n1, Node* n2) { return *n1 == *n2; });
}

bool Json::Object::operator==(Node const& that) const {
    using kv_t = Json::nodepair_t;
    if (this == &that) { return true; }
    if (that.type() != Type::OBJECT) { return false; }
    Json::nodemap_t& that_map = ((Object*)&that)->map;
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
    (root = that.root)->refcnt++;
}

/** Move constructor. */
Json::Json(Json&& that) noexcept {
    root = std::move(that.root);
    that.root = nullptr;
}

Json::Json(std::initializer_list<Json> args) {
    (root = new Array())->refcnt++;
    for (auto arg : args) { *this << arg; }
}

/** Copy assignment */
Json& Json::operator=(Json const& that) {
    root->unref();
    (root = that.root)->refcnt++;
    return *this;
}

/** Move assignment */
Json& Json::operator=(Json&& that) noexcept {
    root->unref();
    root = std::move(that.root);
    that.root = nullptr;
    return *this;
}

Json::~Json() {
    if (root != nullptr) { root->unref(); }
}

Json::Object* Json::mkobject() {
    if (root->type() == Type::JSNULL) {
        root = new Object();
        root->refcnt++;
    }
    if (root->type() != Type::OBJECT)
        imread_raise(JSONUseError,
            "Json::mkobject() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    return (Object*)root;
}
Json::Object* Json::mkobject() const {
    if (root->type() != Type::OBJECT)
        imread_raise(JSONUseError,
            "Json::mkobject() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    return (Object*)root;
}

Json& Json::set(std::string const& key, Json const& value) {
    if (value.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    mkobject()->set(key, value.root);
    return *this;
}

Json::Array* Json::mkarray() {
    if (root->type() == Type::JSNULL) {
        root = new Array();
        root->refcnt++;
    }
    if (root->type() != Type::ARRAY)
        imread_raise(JSONUseError,
            "Json::mkarray() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::ARRAY)");
    return (Array*)root;
}
Json::Array* Json::mkarray() const {
    if (root->type() != Type::ARRAY)
        imread_raise(JSONUseError,
            "Json::mkarray() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::ARRAY)");
    return (Array*)root;
}

Json& Json::operator<<(Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    mkarray()->add(that.root);
    return *this;
}

Json& Json::insert(int idx, Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency"); }
    mkarray()->ins(idx, that.root);
    return *this;
}

Json& Json::replace(int idx, Json const& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
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
        imread_raise(JSONUseError,
            "cyclic dependency");
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
        imread_raise(JSONUseError,
            "cyclic dependency"); }
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
         ++it) { if (other.root == *it) { return idx; }
                 ++idx; }
    return -1;
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
    if (host->type() == Type::OBJECT) {
        ((Object*)host)->set(key, that.root);
    } else if (host->type() == Type::ARRAY) {
        ((Array*)host)->repl(kidx, that.root);
    } else {
        imread_raise(JSONLogicError,
            "Property::operator=(Json) assignment logic error:",
         FF("\tAssignment attempt made on LHS object of Type::%s", host->typestr()),
            "\tJson LHS object (assignee) should be Type::OBJECT or Type::ARRAY");
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

std::size_t Json::size() const {
    if (root->type() == Type::ARRAY)  { return ((Array*)root)->list.size(); }
    if (root->type() == Type::OBJECT) { return ((Object*)root)->map.size(); }
    if (root->type() == Type::STRING) { return ((String*)root)->value.size(); }
    imread_raise(JSONUseError,
        "Json::size() method not applicable",
     FF("root->type() == Type::%s", root->typestr()),
        "(Requires {Type::OBJECT | Type::ARRAY | Type::STRING})");
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
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    Json out = Json{};
    auto const& a = static_cast<Object*>(root)->map;
    auto const& b = static_cast<Object*>(other.root)->map;
    auto op = out.mkobject();
    std::for_each(b.begin(), b.end(),
              [&](auto const& pair) {
        op->set(*pair.first, pair.second);
    });
    std::for_each(a.begin(), a.end(),
              [&](auto const& pair) {
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
    out.root->refcnt--;                     /// back to reality (>= 1)
    return out;
}

Json Json::pop(std::string const& key, Json const& default_value) {
    Node* node = mkobject()->pop(key);      /// refcount unaffected (>= 1)
    if (!node) { return default_value; }
    Json out(node);                         /// constructor increments refcount (>= 2)
    out.root->refcnt--;                     /// back to reality (>= 1)
    return out;
}

Json::Property::Property(Node* node, std::string const& key)
    :host(node), key(key), kidx(-1)
    {
        if (node->type() != Type::OBJECT) {
            imread_raise(JSONUseError,
                "Json::Property::Property(node, key) method not applicable",
             FF("\tnode->type() == Type::%s", node->typestr()),
                "\t(Requires Type::OBJECT)");
        }
    }

Json::Property::Property(Node* node, int idx)
    :host(node), key(""), kidx(idx)
    {
        if (node->type() != Type::ARRAY) {
            imread_raise(JSONUseError,
                "Json::Property::Property(node, idx) method not applicable",
             FF("\tnode->type() == Type::%s", node->typestr()),
                "(Requires Type::ARRAY)");
        }
    }

Json Json::Property::target() const {
    if (host->type() == Type::OBJECT) { return ((Object*)host)->get(key); }
    if (host->type() == Type::ARRAY)  { return ((Array*)host)->list.at(kidx); }
    imread_raise(JSONLogicError,
        "Property::operator Json() conversion-operator logic error:",
     FF("\tConverstion attempt made on Property object of Type::%s", host->typestr()),
        "\tConverted Property object should be Type::OBJECT or Type::ARRAY");
}

detail::stringvec_t Json::keys() const {
    Object* op = mkobject();
    detail::stringvec_t out;
    for (auto const& it : op->map) {
        out.push_back(*it.first);
    }
    return out;
}

detail::stringvec_t Json::allkeys() {
    detail::stringvec_t out;
    for (auto const& key : keyset) {
        out.push_back(key);
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

bool Json::String::operator==(Node const& that) const {
    return this == &that || (
            that.type() == Type::STRING &&
            value == ((String*)&that)->value);
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

void Json::Object::traverse(Json::traverser_t traverser) const {
    for (auto it : map) { it.second->traverse(traverser); }
}

void Json::Object::traverse(Json::named_traverser_t named_traverser,
                            char const* name) const {
    for (auto const& it : map) {
        it.second->traverse(
            named_traverser,
            it.first->c_str());
    }
}

void Json::Array::traverse(Json::traverser_t traverser) const {
    for (auto it : list) { it->traverse(traverser); }
}

void Json::Array::traverse(Json::named_traverser_t named_traverser,
                           char const* name) const {
    std::size_t idx = 0;
    for (auto it : list) {
        it->traverse(
            named_traverser,
            std::to_string(idx).c_str());
        ++idx;
    }
}

void Json::traverse(Json::traverser_t traverser) const {
    root->traverse(traverser);
}

void Json::traverse(Json::named_traverser_t named_traverser) const {
    root->traverse(named_traverser, "root");
}

void Json::Object::print(std::ostream& out) const {
    out << '{';
    ++level;
    bool comma = false;
    for (auto const& it : map) {
        if (comma)  { out << ','; }
        if (indent) { out << '\n'
                          << std::string(indent*level, ' '); }
        detail::escape(out, *it.first);
        out << ':';
        if (indent) { out << ' '; }
        it.second->print(out);
        comma = true;
    }
    --level;
    out << '}';
}

void Json::Array::print(std::ostream& out) const {
    out << '[';
    ++level;
    bool comma = false;
    for (Node const* it : list) {
        if (comma)  { out << ','; }
        if (indent) { out << '\n'
                          << std::string(indent*level, ' '); }
        it->print(out);
        comma = true;
    }
    --level;
    out << ']';
}

Json::Object::~Object() {
    for (auto const& it : map) {
        Node* np = it.second;
        np->unref();
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
    imread_assert(v != nullptr,
                  "Json::Object::set(k, v) called with v == nullptr");
    auto kit = keyset.insert(keyset.begin(), k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        Node* np = it->second;
        np->unref();
        it->second = v;
    } else {
        map[&*kit] = v;
    }
    v->refcnt++;
}

bool Json::Object::del(std::string const& k) {
    /// Unrefs Node* for key, erases it from the map and keyset
    auto kit = keyset.find(k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        Node* np = it->second;
        np->unref();
        map.erase(&*kit);
        keyset.erase(kit);
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
        keyset.erase(kit);
        return np;
    }
    return nullptr;
}

Json::Array::~Array() {
    for (Node* it : list) { it->unref(); }
    list.clear();
}

void Json::Array::add(Node* v) {
    imread_assert(v != nullptr,
                  "Json::Array::add(v) called with v == nullptr");
    list.push_back(v);
    v->refcnt++;
}

void Json::Array::pop() {
    Node* v = (Node*)list.back();
    v->unref();
    list.pop_back();
}

/** Inserts given Node* before index. */
void Json::Array::ins(int idx, Node* v) {
    imread_assert(v != nullptr,
                  "Json::Array::ins(idx, v) called with v == nullptr");
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    list.insert(list.begin() + idx, v);
    v->refcnt++;
}

void Json::Array::del(int idx) {
    if (idx < 0) { idx = list.size(); }
    if (idx < 0 || idx >= (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    Node* v = list.at(idx);
    v->unref();
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
    u->unref();
    list[idx] = v;
    v->refcnt++;
}

std::ostream& operator<<(std::ostream& out, Json const& json) {
    json.root->print(out);
    return out;
}

std::istream& operator>>(std::istream& in, Json& json) {
    // json.root->unref();
    // Json temp(in);
    // json.root = temp.root;
    // temp.root = nullptr;
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
        root->refcnt++;
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
        root->refcnt++;
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
            set(key, obj);
            in >> c;
            if (c == ',') { continue; }
            in.putback(c);
        }
        throw parse_error("comma or closing bracket expected", in);
    }
    if (std::isdigit(c) || c == '-') {
        in.putback(c);
        root = new Number(in);
        root->refcnt++;
        goto out;
    }
    if (c == '\"' || c == '\'') {
        in.putback(c);
        root = new String(in);
        root->refcnt++;
        goto out;
    }
    word.push_back(c);
    for (int i = 0; i < 3; i++) { word.push_back(in.get()); }
    if (word == "null") {
        root = &Node::null;
        root->refcnt++;
        goto out;
    }
    if (word == "true") {
        root = &Bool::T;
        root->refcnt++;
        goto out;
    }
    if (word == "fals" && in.get() == 'e') {
        root = &Bool::F;
        root->refcnt++;
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

std::string Json::format() const {
    mute.lock();
    std::ostringstream out;
    out << *this;
    std::string outstr(out.str());
    mute.unlock();
    return outstr;
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

std::size_t Json::hash(std::size_t H) const {
    hash::rehash<std::string>(H, format());
    return H;
}

Json::operator std::string() const {
    if (root->type() == Type::STRING) { return ((String*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator long double() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator double() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator float() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator int() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator long() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator long long() const {
    if (root->type() == Type::NUMBER) { return ((Number*)root)->value; }
    imread_raise_default(JSONBadCast);
}

Json::operator uint8_t() const {
    if (root->type() == Type::NUMBER) { return uint8_t(((Number*)root)->value); }
    imread_raise_default(JSONBadCast);
}

Json::operator uint16_t() const {
    if (root->type() == Type::NUMBER) { return uint16_t(((Number*)root)->value); }
    imread_raise_default(JSONBadCast);
}

Json::operator int8_t() const {
    if (root->type() == Type::NUMBER) { return int8_t(((Number*)root)->value); }
    imread_raise_default(JSONBadCast);
}

Json::operator int16_t() const {
    if (root->type() == Type::NUMBER) { return int16_t(((Number*)root)->value); }
    imread_raise_default(JSONBadCast);
}

Json::operator bool() const {
    switch (root->type()) {
        case Type::BOOLEAN: return root == &Bool::T;
        case Type::NUMBER:  return static_cast<bool>(int(((Number*)root)->value));
        case Type::STRING:  return static_cast<bool>(((String*)root)->value.size());
        case Type::JSNULL:  return false;
        default:            imread_raise_default(JSONBadCast);
    }
    imread_raise_default(JSONBadCast);
}

bool Json::operator==(Json const& that) const {
    if (root == that.root) { return true; }
    return *root == *that.root;
}

template <>
filesystem::path Json::cast<filesystem::path>(std::string const& key) const {
    return filesystem::path(static_cast<std::string>(get(key)));
}
template <>
const char* Json::cast<const char*>(std::string const& key) const {
    return static_cast<std::string>(get(key)).c_str();
}
template <>
char* Json::cast<char*>(std::string const& key) const {
    return const_cast<char*>(static_cast<std::string>(get(key)).c_str());
}
