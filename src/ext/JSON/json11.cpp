/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#include <libimread/ext/JSON/json11.h>
#include <assert.h>
#include <cmath>
#include <cfloat>
#include <climits>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;

Json::Node Json::Node::null(1);
Json::Node Json::Node::undefined(1);
Json Json::null;
Json Json::undefined(&Node::undefined);
Json::Bool Json::Bool::T(true);
Json::Bool Json::Bool::F(false);
set<string> Json::keyset;
int Json::indent;
int Json::level;

static unsigned currpos(istream& in, unsigned* pos) {
    unsigned curr = in.tellg();
    if (pos != nullptr)
        *pos = 0;
    in.seekg(0);   // rewind
    if (in.bad())
        return 0;
    unsigned count = 0, line = 1, col = 1;
    while (!in.eof() && !in.fail() && ++count < curr) {
        if (in.get() == '\n') {
            ++line;
            col = 1;
        } else
            ++col;
    }
    if (pos != nullptr)
        *pos = col;
    return line;
}

Json::parse_error::parse_error(const char* msg, std::istream& in) : std::runtime_error(msg) {
    line = currpos(in, &col);
}

// Node and helper classes

Json::Node::Node(unsigned init) {
    refcnt = init;
}

Json::Node::~Node() {
    assert(this == &null || this == &undefined || this == &Bool::T || this == &Bool::F || refcnt == 0);
}

void Json::Node::unref() {
    if (this == &null || this == &undefined || this == &Bool::T || this == &Bool::F )
        return;
    assert(refcnt > 0);
    if (--refcnt == 0)
        delete this;
}

bool Json::Array::operator == (const Node& that) const {
    if (this == &that)
        return true;
    if (that.type() != Type::ARRAY)
        return false;
    vector<Node*>& that_list = ((Array*)&that)->list;
    return equal(list.begin(), list.end(), that_list.begin(),
            [](Node* n1, Node* n2){ return *n1 == *n2; });
}

bool Json::Object::operator == (const Node& that) const {
    if (this == &that)
        return true;
    if (that.type() != Type::OBJECT)
        return false;
    std::map<const string*, Node*>& that_map = ((Object*)&that)->map;
    typedef pair<const string*, Node*> kv;
    return equal(map.begin(), map.end(), that_map.begin(),
            [](kv p, kv q){ return *p.first == *q.first && *p.second == *q.second; });
}

bool Json::Number::operator == (const Node& that) const {
    if (this == &that)
        return true;
    if (that.type() != Type::NUMBER)
        return false;
    Number& numb = *(Number*)&that;
    if (fabs(value) < LDBL_EPSILON)
        return fabs(numb.value) < LDBL_EPSILON;
    long double delta = fabs((value - numb.value)/value);
    int digs = max(prec, numb.prec);
    return delta < pow(10, -digs);
}

bool Json::Array::contains(const Node* that) const {
    if (this == that)
        return true;
    for (Node* it : list) {
        if (it->contains(that))
            return true;
    }
    return false;
}

bool Json::Object::contains(const Node* that) const {
    if (this == that)
        return true;
    for (auto it : map) {
        if (it.second->contains(that))
            return true;
    }
    return false;
}

/** Copy constructor. */
Json::Json(const Json& that) {
	(root = that.root)->refcnt++;
}

/** Move constructor. */
Json::Json(Json&& that) {
	root = that.root;
	that.root = nullptr;
}

Json::Json(std::initializer_list<Json> args) {
    (root = new Array())->refcnt++;
	for (auto arg : args)
		*this << arg;
}

/** Copy assignment */
Json& Json::operator = (const Json& that) {
    root->unref();
    (root = that.root)->refcnt++;
    return *this;
}

/** Move assignment */
Json& Json::operator = (Json&& that) {
    root->unref();
    root = that.root;
    that.root = nullptr;
    return *this;
}

Json::~Json() {
    if (root != nullptr)
        root->unref();
}

Json::Object* Json::mkobject() {
    if (root->type() == Type::JSNULL) {
        root = new Object();
        root->refcnt++;
    }
    if (root->type() != Type::OBJECT)
        throw use_error("method not applicable");
    return (Object*)root;
}

Json& Json::set(string key, const Json& val) {
    assert(val.root != nullptr);
    if (val.root->contains(root))
        throw use_error("cyclic dependency");
    mkobject()->set(key, val.root);
    return *this;
}

Json::Array* Json::mkarray() {
    if (root->type() == Type::JSNULL) {
        root = new Array();
        root->refcnt++;
    }
    if (root->type() != Type::ARRAY)
        throw use_error("method not applicable");
    return (Array*)root;
}

Json& Json::operator << (const Json& that) {
    if (that.root->contains(root))
        throw use_error("cyclic dependency");
    mkarray()->add(that.root);
    return *this;
}

void Json::insert(int index, const Json& that) {
    if (that.root->contains(root))
        throw use_error("cyclic dependency");
    mkarray()->ins(index, that.root);
}

Json& Json::replace(int index, const Json& that) {
    if (that.root->contains(root))
        throw use_error("cyclic dependency");
    mkarray()->repl(index, that.root);
    return *this;
}

void Json::erase(int index) {
    mkarray()->del(index);
}

Json Json::Property::operator = (const Json& that) {
    if (host->type() == Type::OBJECT)
        ((Object*)host)->set(key, that.root);
    else if (host->type() == Type::ARRAY)
        ((Array*)host)->repl(index, that.root);
    else
        throw logic_error("Property::operator =");
    return target();
}

Json Json::Property::operator = (const Property& that) {
    return (*this = that.target());
}

Json::Property Json::operator [] (int index) {
    return Property(mkarray(), index);
}

Json::Property Json::operator [] (const string& key) {
    return Property(mkobject(), key);
}

size_t Json::size() const {
    if (root->type() == Type::ARRAY)
        return ((Array*)root)->list.size();
    if (root->type() == Type::OBJECT)
        return ((Object*)root)->map.size();
    throw use_error("method not applicable");
}

Json Json::get(const string& key) const {
    if (root->type() != Type::OBJECT)
        throw use_error("method not applicable");
    Node* n = ((Object*)root)->get(key);
    return n == nullptr ? undefined : Json(n);
}

bool Json::has(const string& key) const {
    if (root->type() != Type::OBJECT)
        throw use_error("method not applicable");
    auto kp = keyset.find(key);
    if (kp == keyset.end())
        return false;
    Object* obj = (Object*)root;
    auto it = obj->map.find(&*kp);
    return it != obj->map.end();
}

Json::Property::Property(Node* node, const string& key) : host(node) {
    if (node->type() != Type::OBJECT)
        throw use_error("method not applicable");
    this->key = key;
    index = -1;
}

Json::Property::Property(Node* node, int index) : host(node) {
    if (node->type() != Type::ARRAY)
        throw use_error("method not applicable");
    key = "";
    this->index = index;
}

Json Json::Property::target() const {
    if (host->type() == Type::OBJECT)
        return ((Object*)host)->get(key);
    if (host->type() == Type::ARRAY)
        return ((Array*)host)->list.at(index);
    throw logic_error("Property::operator Json()");
}

vector<string> Json::keys() {
    if (root->type() != Type::OBJECT)
        throw use_error("method not applicable");
    Object* op = (Object*)root;
    vector<string> ret;
    for (auto it : op->map)
        ret.push_back(*it.first);
    return ret;
}

bool Json::String::operator == (const Node& that) const {
    return this == &that ||
            (that.type() == Type::STRING && value == ((String*)&that)->value);
}

void Json::Bool::print(ostream& out) const {
    out << (this == &Bool::T ? "true" : "false");
}

void Json::Number::print(ostream& out) const {
    if (prec >= 0)
        out << setprecision(prec);
    out << value;
}

static void escape(ostream& out, const string& str) {
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

void Json::String::print(ostream& out) const {
    escape(out, value);
}

void Json::Object::traverse(void (*f)(const Node*)) const {
    for (auto it : map)
        f(it.second);
}

void Json::Array::traverse(void (*f)(const Node*)) const {
    for (auto it : list)
        f(it);
}

void Json::Object::print(ostream& out) const {
    out << '{';
    ++level;
    bool comma = false;
    for (auto it : map) {
        if (comma)
            out << ',';
        if (indent)
            out << '\n' << string(indent*level, ' ');
        escape(out, *it.first);
        out << ':';
        if (indent)
            out << ' ';
        it.second->print(out);
        comma = true;
    }
    --level;
    out << '}';
}

void Json::Array::print(ostream& out) const {
    out << '[';
    ++level;
    bool comma = false;
    for (const Node* it : list) {
        if (comma)
            out << ',';
        if (indent)
            out << '\n' << string(indent*level, ' ');
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

Json::Node* Json::Object::get(const string& key) const {
    auto kp = keyset.find(key);
    if (kp == keyset.end())
        return nullptr;
    auto it = map.find(&*kp);
    if (it == map.end())
        return nullptr;
    return it->second;

}

void Json::Object::set(const string& k, Node* v) {
    assert(v != nullptr);
    auto kit = keyset.insert(keyset.begin(), k);
    auto it = map.find(&*kit);
    if (it != map.end()) {
        Node* np = it->second;
        np->unref();
        it->second = v;
    } else
        map[&*kit] = v;
    v->refcnt++;
}

Json::Array::~Array() {
    for (Node* it : list)
        it->unref();
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
    if (index < 0)
        index += list.size();
    if (index < 0 || index > (int)list.size())
        throw out_of_range("index out of range");
    list.insert(list.begin() + index, v);
    v->refcnt++;
}

void Json::Array::del(int index) {
    if (index < 0)
        index += list.size();
    Node* v = list.at(index);
    v->unref();
    list.erase(list.begin() + index);
}

void Json::Array::repl(int index, Node* v) {
    if (index < 0)
        index += list.size();
    Node* u = list.at(index);
    u->unref();
    list[index] = v;
    v->refcnt++;
}

ostream& operator << (ostream& out, const Json& json) {
    json.root->print(out);
    return out;
}

istream& operator >> (istream& in, Json& json) {
    json.root->unref();
    Json temp(in);
    json.root = temp.root;
    temp.root = nullptr;
    return in;
}

Json::String::String(istream& in) {
    int quote = in.get();
    while (!in.eof()) {
        int c = in.get();
        if (c == char_traits<char>::eof())
            throw parse_error("unterminated string", in);
        if (c == quote)
            return;
        if (c == '\\') {
            c = in.get();
            if (c == quote || c == '\\' || c == '/')
                value.push_back(c);
            else if (c == 'n')
                value.push_back('\n');
            else if (c == 't')
                value.push_back('\t');
            else if (c == 'r')
                value.push_back('\r');
            else if (c == 'b')
                value.push_back('\b');
            else if (c == 'f')
                value.push_back('\f');
            else if (c == 'u') {
                unsigned w = 0;
                for (int i = 0; i < 4; i++) {
                    if (!isxdigit(c = toupper(in.get())))
                        throw parse_error("not a hex digit", in);
                    w = (w << 4) | (isdigit(c) ? c - '0' : c - 'A' + 10);
                }
                // garbage in, garbage out
                if (w <= 0x7f)
                    value.push_back(w);
                else if (w <= 0x07ff) {
                    value.push_back(0xc0 | ((w >> 6) & 0x1f));
                    value.push_back(0x80 | (w & 0x3f));
                } else {
                    value.push_back(0xe0 | ((w >> 12) & 0x0f));
                    value.push_back(0x80 | ((w >> 6) & 0x3f));
                    value.push_back(0x80 | (w & 0x3f));
                }
            }
            else
                throw parse_error("illegal backslash escape", in);
            continue;
        }
        if (iscntrl(c))
            throw parse_error("control character in string", in);
        value.push_back(c);
    }
}

Json::Number::Number(istream& in) {
    char buf[128];
    const char* end = buf+126;
    char* p = buf;
    char c;
    bool leading = true;
    while ((isdigit(c = in.get()) || (leading && c == '-')) && p < end) {
        *p++ = c;
        leading = false;
    }
    prec = p - buf;
    if (c == '.' && p < end) {
        *p++ = c;
        while (isdigit(c = in.get()) && p < end)
            *p++ = c;
        prec = p - buf - 1;
    }
    if ((c == 'e' || c == 'E') && p < end) {
        *p++ = c;
        leading = true;
        while ((isdigit(c = in.get()) || (leading && (c == '-' || c == '+'))) && p < end) {
            *p++ = c;
            leading = false;
        }
    }
    *p = 0;
    in.putback(c);
    char* eptr = nullptr;
    long double num = strtold(buf, &eptr);
    if (eptr != p)
        throw parse_error("illegal number format", in);
    value = num;
}

Json::Json(istream& in, bool full) {
    char c;
    string word;
    root = nullptr;
    if (!(in >> c))
        goto out;
    if (c == '[') {
        root = new Array();
        root->refcnt++;
        while (in >> c) {
            if (c == ']')
                goto out;
            in.putback(c);
            Json elem(in, false);
            *this << elem;
            in >> c;
            if (c == ',')
                continue;
            in.putback(c);
        }
        throw parse_error("comma or closing bracket expected", in);
    }
    if (c == '{') {
        root = new Object();
        root->refcnt++;
        while (in >> c) {
            if (c == '}')
                goto out;
            in.putback(c);
            Json key(in, false);
            if (key.root->type() != Type::STRING)
                throw parse_error("a string expected", in);
            in >> c;
            if (c != ':')
                throw parse_error("a colon expected", in);
            Json obj(in, false);
            set(key, obj);
            in >> c;
            if (c == ',')
                continue;
            in.putback(c);
        }
        throw parse_error("comma or closing bracket expected", in);
    }
    if (isdigit(c) || c == '-') {
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
    for (int i = 0; i < 3; i++)
        word.push_back(in.get());
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
        if (in.peek() == char_traits<char>::eof())
            return;
        while (isspace(in.get()))
            /* skip */;
        if (in.eof())
            return;
        throw parse_error("excess text not parsed", in);
    }
}

string Json::format() {
    ostringstream is("");
    is << *this;
    return is.str();
}

Json Json::parse(const string& str) {
    istringstream is(str);
    Json parsed(is);
    if (is.peek() == char_traits<char>::eof())
        return parsed;
    while (isspace(is.get()))
        /* skip */;
    if (is.eof())
        return parsed;
    throw parse_error("json format error", is);
}

Json::operator std::string() const {
    if (root->type() == Type::STRING)
        return ((String*)root)->value;
    throw bad_cast();
}

Json::operator long double() const {
    if (root->type() == Type::NUMBER)
        return ((Number*)root)->value;
    throw bad_cast();
}

Json::operator double() const {
    if (root->type() == Type::NUMBER)
        return ((Number*)root)->value;
    throw bad_cast();
}

Json::operator float() const {
    if (root->type() == Type::NUMBER) {
        return ((Number*)root)->value;
    }
    throw bad_cast();
}

Json::operator int() const {
    if (root->type() == Type::NUMBER)
        return ((Number*)root)->value;
    throw bad_cast();
}

Json::operator long() const {
    if (root->type() == Type::NUMBER)
        return ((Number*)root)->value;
    throw bad_cast();
}

Json::operator long long() const {
    if (root->type() == Type::NUMBER)
        return ((Number*)root)->value;
    throw bad_cast();
}

// Json::operator uint8_t() const {
//     if (root->type() == Type::NUMBER)
//         return uint8_t(((Number*)root)->value);
//     throw bad_cast();
// }
//
// Json::operator uint16_t() const {
//     if (root->type() == Type::NUMBER)
//         return uint16_t(((Number*)root)->value);
//     throw bad_cast();
// }
//
// Json::operator int8_t() const {
//     if (root->type() == Type::NUMBER)
//         return int8_t(((Number*)root)->value);
//     throw bad_cast();
// }
//
// Json::operator int16_t() const {
//     if (root->type() == Type::NUMBER)
//         return int16_t(((Number*)root)->value);
//     throw bad_cast();
// }

Json::operator unsigned char() const {
    if (root->type() == Type::NUMBER)
        return (unsigned char)((Number*)root)->value;
    throw bad_cast();
}
Json::operator char() const {
    if (root->type() == Type::NUMBER)
        return char(((Number*)root)->value);
    throw bad_cast();
}

Json::operator bool() const {
    if (root->type() == Type::BOOLEAN)
        return root == &Bool::T;
    throw bad_cast();
}

bool Json::operator == (const Json& that) const {
    if (root == that.root)
        return true;
    return *root == *that.root;
}

