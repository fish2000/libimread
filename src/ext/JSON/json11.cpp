/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#include <libimread/ext/JSON/json11.h>
#include <libimread/errors.hh>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <climits>
#include <sstream>
#include <iomanip>
#include <algorithm>

Json::Node Json::Node::null(1);
Json::Node Json::Node::undefined(1);
Json Json::null;
Json Json::undefined(&Node::undefined);
Json::Bool Json::Bool::T(true);
Json::Bool Json::Bool::F(false);
std::set<std::string> Json::keyset;
int Json::indent;
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
            case Type::SCHEMA:  return "SCHEMA";
        }
        imread_raise_default(JSONOutOfRange);
    }
    
}

namespace detail {
    
    unsigned currpos(std::istream& in, unsigned* pos) {
        unsigned curr = in.tellg();
        if (pos != nullptr) { *pos = 0; }
        in.seekg(0); // rewind
        if (in.bad()) { return 0; }
        unsigned count = 0, line = 1, col = 1;
        while (!in.eof() && !in.fail() && ++count < curr) {
            if (in.get() == '\n') {
                ++line;
                col = 1;
            } else { ++col; }
        }
        if (pos != nullptr) { *pos = col; }
        return line;
    }
    
    void escape(std::ostream& out, const std::string& str) {
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
            }
        }
        out << '"';
    }
    
}

Json::parse_error::parse_error(const char* msg, std::istream& in)
    :im::JSONParseError(msg)
    {
        line = detail::currpos(in, &col);
    }

Json::parse_error::parse_error(const std::string& msg, std::istream& in)
    :im::JSONParseError(msg)
    {
        line = detail::currpos(in, &col);
    }

/// Node and helper classes
Json::Node::Node(unsigned init) {
    refcnt = init;
}

Json::Node::~Node() {
    assert(this == &null || this == &undefined || this == &Bool::T || this == &Bool::F || refcnt == 0);
}

void Json::Node::unref() {
    if (this == &null || this == &undefined || this == &Bool::T || this == &Bool::F) {
        return;
    }
    assert(refcnt > 0);
    if (--refcnt == 0) { delete this; }
}

bool Json::Array::operator==(const Node& that) const {
    if (this == &that) { return true; }
    if (that.type() != Type::ARRAY) { return false; }
    std::vector<Node*>& that_list = ((Array*)&that)->list;
    return std::equal(list.begin(), list.end(),
                 that_list.begin(),
                   [](Node* n1, Node* n2) { return *n1 == *n2; });
}

bool Json::Object::operator==(const Node& that) const {
    using kv = std::pair<const std::string*, Node*>;
    if (this == &that) { return true; }
    if (that.type() != Type::OBJECT) { return false; }
    std::map<const std::string*, Node*>& that_map = ((Object*)&that)->map;
    return std::equal(map.begin(), map.end(),
                 that_map.begin(),
                   [](kv p, kv q) { return *p.first == *q.first &&
                                           *p.second == *q.second; });
}

bool Json::Number::operator==(const Node& that) const {
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

bool Json::Array::contains(const Node* that) const {
    if (this == that) { return true; }
    for (Node* it : list) {
        if (it->contains(that)) { return true; }
    }
    return false;
}

bool Json::Object::contains(const Node* that) const {
    if (this == that) { return true; }
    for (auto it : map) {
        if (it.second->contains(that)) { return true; }
    }
    return false;
}

/** Copy constructor. */
Json::Json(const Json& that) {
    (root = that.root)->refcnt++;
}

/** Move constructor. */
Json::Json(Json&& that) noexcept {
    root = that.root;
    that.root = nullptr;
}

Json::Json(std::initializer_list<Json> args) {
    (root = new Array())->refcnt++;
    for (auto arg : args) { *this << arg; }
}

/** Copy assignment */
Json& Json::operator=(const Json& that) {
    root->unref();
    (root = that.root)->refcnt++;
    return *this;
}

/** Move assignment */
Json& Json::operator=(Json&& that) noexcept {
    root->unref();
    root = that.root;
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

Json& Json::set(std::string key, const Json& val) {
    assert(val.root != nullptr);
    if (val.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    mkobject()->set(key, val.root);
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

Json& Json::operator<<(const Json& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    mkarray()->add(that.root);
    return *this;
}

Json& Json::insert(int index, const Json& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency"); }
    mkarray()->ins(index, that.root);
    return *this;
}

Json& Json::replace(int index, const Json& that) {
    if (that.root->contains(root)) {
        imread_raise(JSONUseError,
            "cyclic dependency");
    }
    mkarray()->repl(index, that.root);
    return *this;
}

Json& Json::erase(int index) {
    mkarray()->del(index);
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
    mkarray()->ins(-1, other.root);
    return *this;
}

int Json::index(Json const& other) const {
    if (root->type() != Type::ARRAY)
        imread_raise(JSONUseError,
            "Json::index(Json const&) method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::ARRAY)");
    auto const& arlist = static_cast<Array*>(root)->list;
    if (!root->contains(other.root)) { return -1; }
    int idx = 0;
    for (auto it = arlist.begin();
         it != arlist.end();
         ++it) { if (other.root == *it) { return idx; }
                 ++idx; }
    return -1;
}

Json Json::pop() {
    auto const& arlist = mkarray()->list;
    if (arlist.empty()) { return null; }
    Json out(arlist.back());
    erase(-1);
    return out;
}


Json Json::Property::operator=(const Json& that) {
    if (host->type() == Type::OBJECT) {
        ((Object*)host)->set(key, that.root);
    } else if (host->type() == Type::ARRAY) {
        ((Array*)host)->repl(index, that.root);
    } else {
        imread_raise(JSONLogicError,
            "Property::operator=(Json) assignment logic error:",
         FF("\tAssignment attempt made on LHS object of Type::%s", host->typestr()),
            "\tJson LHS object (assignee) should be Type::OBJECT or Type::ARRAY");
    }
    return target();
}

Json Json::Property::operator=(const Property& that) {
    return (*this = that.target());
}

Json::Property Json::operator[](int index) {
    return Property(mkarray(), index);
}

Json::Property Json::operator[](const std::string& key) {
    return Property(mkobject(), key);
}

std::size_t Json::size() const {
    if (root->type() == Type::ARRAY) { return ((Array*)root)->list.size(); }
    if (root->type() == Type::OBJECT) { return ((Object*)root)->map.size(); }
    if (root->type() == Type::STRING) { return ((String*)root)->value.size(); }
    imread_raise(JSONUseError,
        "Json::size() method not applicable",
     FF("root->type() == Type::%s", root->typestr()),
        "(Requires {Type::OBJECT | Type::ARRAY | Type::STRING})");
}

Json Json::get(const std::string& key) const {
    if (root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::get(key) method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    Node* n = ((Object*)root)->get(key);
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
            "Json::get(key) method not applicable",
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

bool Json::has(const std::string& key) const {
    if (root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::has(key) method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    auto kp = keyset.find(key);
    if (kp == keyset.end()) { return false; }
    Object* obj = (Object*)root;
    auto it = obj->map.find(&*kp);
    return it != obj->map.end();
}

Json::Property::Property(Node* node, const std::string& key) : host(node) {
    if (node->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::Property::Property(node, key) method not applicable",
         FF("\tnode->type() == Type::%s", node->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    this->key = key;
    index = -1;
}

Json::Property::Property(Node* node, int index) : host(node) {
    if (node->type() != Type::ARRAY) {
        imread_raise(JSONUseError,
            "Json::Property::Property(node, idx) method not applicable",
         FF("\tnode->type() == Type::%s", node->typestr()),
            "(Requires Type::ARRAY)");
    }
    key = "";
    this->index = index;
}

Json Json::Property::target() const {
    if (host->type() == Type::OBJECT) { return ((Object*)host)->get(key); }
    if (host->type() == Type::ARRAY) { return ((Array*)host)->list.at(index); }
    imread_raise(JSONLogicError,
        "Property::operator Json() conversion-operator logic error:",
     FF("\tConverstion attempt made on Property object of Type::%s", host->typestr()),
        "\tConverted Property object should be Type::OBJECT or Type::ARRAY");
}

std::vector<std::string> Json::keys() {
    if (root->type() != Type::OBJECT) {
        imread_raise(JSONUseError,
            "Json::keys() method not applicable",
         FF("\troot->type() == Type::%s", root->typestr()),
            "\t(Requires Type::OBJECT)");
    }
    Object* op = (Object*)root;
    std::vector<std::string> ret;
    for (auto it : op->map) { ret.push_back(*it.first); }
    return ret;
}

bool Json::String::operator==(const Node& that) const {
    return this == &that || (
            that.type() == Type::STRING &&
            value == ((String*)&that)->value);
}

void Json::Bool::print(std::ostream& out) const {
    out << (this == &Bool::T ? "true" : "false");
}

void Json::Number::print(std::ostream& out) const {
    if (prec >= 0) { out << std::setprecision(prec); }
    out << value;
}

void Json::String::print(std::ostream& out) const {
    detail::escape(out, value);
}

void Json::Object::traverse(void (*f)(const JSONNode*)) const {
    //for (auto it : map) { f(it.second); }
    for (auto it : map) { it.second->traverse(f); }
}

void Json::Array::traverse(void (*f)(const JSONNode*)) const {
    //for (auto it : list) { f(it); }
    for (auto it : list) { it->traverse(f); }
}

void Json::Object::print(std::ostream& out) const {
    out << '{';
    ++level;
    bool comma = false;
    for (auto it : map) {
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
    for (const Node* it : list) {
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
    for (auto it : map) {
        Node* np = it.second;
        np->unref();
    }
    map.clear();
}

Json::Node* Json::Object::get(const std::string& key) const {
    auto kp = keyset.find(key);
    if (kp == keyset.end()) { return nullptr; }
    auto it = map.find(&*kp);
    if (it == map.end()) { return nullptr; }
    return it->second;
}

void Json::Object::set(const std::string& k, Node* v) {
    assert(v != nullptr);
    auto kit = keyset.insert(keyset.begin(), k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        Node* np = it->second;
        np->unref();
        it->second = v;
    } else { map[&*kit] = v; }
    v->refcnt++;
}

Json::Array::~Array() {
    for (Node* it : list) { it->unref(); }
    list.clear();
}

void Json::Array::add(Node* v) {
    assert(v != nullptr);
    list.push_back(v);
    v->refcnt++;
}

/** Inserts given Node* before index. */
void Json::Array::ins(int index, Node* v) {
    assert(v != nullptr);
    // if (index < 0) { index += list.size(); }
    if (index < 0) { index = 0; }
    if (index < 0 || index > (int)list.size()) {
        imread_raise_default(JSONOutOfRange);
    }
    list.insert(list.begin() + index, v);
    v->refcnt++;
}

void Json::Array::del(int index) {
    if (index < 0) { index += list.size(); }
    Node* v = list.at(index);
    v->unref();
    list.erase(list.begin() + index);
}

void Json::Array::repl(int index, Node* v) {
    if (index < 0) { index += list.size(); }
    Node* u = list.at(index);
    u->unref();
    list[index] = v;
    v->refcnt++;
}

std::ostream& operator<<(std::ostream& out, const Json& json) {
    json.root->print(out);
    return out;
}

std::istream& operator>>(std::istream& in, Json& json) {
    json.root->unref();
    Json temp(in);
    json.root = temp.root;
    temp.root = nullptr;
    return in;
}

Json::String::String(std::istream& in) {
    int quote = in.get();
    while (!in.eof()) {
        int c = in.get();
        if (c == std::char_traits<char>::eof()) {
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
        if (in.peek() == std::char_traits<char>::eof()) { return; }
        while (std::isspace(in.get()))
            /* skip */;
        if (in.eof()) { return; }
        throw parse_error("excess text not parsed", in);
    }
}

std::string Json::format() const {
    std::ostringstream is("");
    is << *this;
    return is.str();
}

Json Json::parse(const std::string& str) {
    std::istringstream is(str);
    Json parsed(is);
    if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
    while (std::isspace(is.get()))
        /* skip */;
    if (is.eof()) { return parsed; }
    throw parse_error("JSON format error", is);
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

bool Json::operator==(const Json& that) const {
    if (root == that.root) { return true; }
    return *root == *that.root;
}

