/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#ifdef WITH_SCHEMA

#include <libimread/ext/JSON/json11.h>
#include <algorithm>
#include <limits>

using namespace std;

struct invalid : exception {
    string msg;
    invalid(const char* msg) { this->msg = msg; }
    invalid(string msg) { this->msg = msg; }
};

Json::Schema::Schema(Node* node) {
    if (node->type() != Type::OBJECT)
        throw use_error("not an object");
    Object* obj = (Object*)node;
    Node* nptr = obj->get("$schema");
    if (nptr != nullptr) {
        if (nptr->type() != Type::STRING)
            throw use_error("$schema: not a string");
        uri = ((String*)nptr)->value;
    }
    // TODO $ref
    nptr = obj->get("type");
    if (nptr == nullptr || nptr->type() != Type::STRING)
        throw use_error("type: not a string");
    s_type = ((String*)nptr)->value;
    //
    nptr = obj->get("enum");
    if (nptr != nullptr) {
        if (nptr->type() != Type::ARRAY)
            throw use_error("enum: not an array");
        s_enum = (Array*)nptr;
        s_enum->refcnt++;
    }
    //
    nptr = obj->get("allOf");
    if (nptr != nullptr) {
        if (nptr->type() != Type::ARRAY)
            throw use_error("allOf: not an array");
        auto src = ((Array*)nptr)->list;
        for (unsigned i = 0; i < src.size(); i++) {
            Schema* sp = new Schema(src[i]);
            allof.push_back(sp);
        }
    }
    nptr = obj->get("anyOf");
    if (nptr != nullptr) {
        if (nptr->type() != Type::ARRAY)
            throw use_error("anyOf: not an array");
        auto list = ((Array*)nptr)->list;
        for (unsigned i = 0; i < list.size(); i++) {
            Schema* sp = new Schema(list[i]);
            anyof.push_back(sp);
        }
    }
    nptr = obj->get("oneOf");
    if (nptr != nullptr) {
        if (nptr->type() != Type::ARRAY)
            throw use_error("oneOf: not an array");
        auto list = ((Array*)nptr)->list;
        for (unsigned i = 0; i < list.size(); i++) {
            Schema* sp = new Schema(list[i]);
            oneof.push_back(sp);
        }
    }
    nptr = obj->get("not");
    if (nptr != nullptr) {
        if (nptr->type() != Type::OBJECT)
            throw use_error("not: not an object");
        s_not = new Schema(nptr);
    }
    nptr = obj->get("definitions");
    if (nptr != nullptr) {
        if (nptr->type() != Type::OBJECT)
            throw use_error("definitions: not an object");
        Object* src = (Object*)nptr;
        defs = new Object();
        for (auto kv : src->map) {
            Schema* sp = new Schema(kv.second);
            defs->set(*kv.first, sp);
        }
        defs->refcnt++;
    }
    nptr = obj->get("default");
    if (nptr != nullptr) {
        (deflt = nptr)->refcnt++;
    }
    //
    if (s_type == "number" || s_type == "integer") {
        nptr = obj->get("maximum");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("maximum: not a number");
            max_num = ((Number*)nptr)->value;
        }
        nptr = obj->get("exclusiveMaximum");
        if (nptr != nullptr) {
            if (nptr->type() != Type::BOOLEAN)
                throw use_error("exclusiveMaximum: not a bool");
            if (max_num == UINT32_MAX)
                throw use_error("exclusiveMaximum: no maximum");
            max_exc = (Bool*)nptr == &Bool::T;
        }
        nptr = obj->get("minimum");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("minimum: not a number");
            min_num = ((Number*)nptr)->value;
        }
        nptr = obj->get("exclusiveMinimum");
        if (nptr != nullptr) {
            if (nptr->type() != Type::BOOLEAN)
                throw use_error("exclusiveMinimum: not a bool");
            if (min_num == -UINT32_MAX)
                throw use_error("exclusiveMinimum: no minimum");
            min_exc = (Bool*)nptr == &Bool::T;
        }
        nptr = obj->get("multipleOf");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("multipleOf: not a number");
            mult_of = ((Number*)nptr)->value;
            if (mult_of <= 0)
                throw use_error("multipleOf: not positive");
        }
    } else if (s_type == "string") {
        nptr = obj->get("maxLength");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("maxLength: not a number");
            max_len = ((Number*)nptr)->value;
            /*
            if (max_len < 0)
                throw use_error("maxLength: is negative");
            */
            if (max_len != ((Number*)nptr)->value)
                throw use_error("maxLength: not an integer");
        }
        nptr = obj->get("minLength");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("minLength: not a number");
            min_len = ((Number*)nptr)->value;
            /*
            if (min_len < 0)
                throw use_error("minLength: is negative");
            */
            if (min_len != ((Number*)nptr)->value)
                throw use_error("minLength: not an integer");
            if (max_len < min_len)
                throw use_error("minLength greater than maxLength");
        }
        nptr = obj->get("pattern");
        if (nptr != nullptr) {
            try {
                if (nptr->type() != Type::STRING)
                    throw use_error("pattern: not a string");
                pattern = new regex(((String*)nptr)->value, regex_constants::ECMAScript);
            } catch (regex_error& ex) {
                throw use_error(string("pattern: ") + ex.what());
            }
        }
    } else if (s_type == "array") {
        nptr = obj->get("items");
        if (nptr != nullptr) {
            if (nptr->type() == Type::OBJECT) {
                item = new Schema(nptr);
            } else if (nptr->type() == Type::ARRAY) {
                for (Node* n : ((Array*)nptr)->list) {
                    Schema* sp = new Schema(n);
                    items.push_back(sp);
                }
            } else
                throw use_error("items: not an object or array");
        }
        nptr = obj->get("additionalItems");
        if (nptr != nullptr) {
            if (nptr->type() == Type::OBJECT)
                add_items = new Schema(nptr);
            else if (nptr->type() == Type::BOOLEAN)
                add_items_bool = (Bool*)nptr == &Bool::T;
            else
                throw use_error("additionalItems: not an object or bool");
        }
        nptr = obj->get("maxItems");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("maxItems: not a number");
            max_len = ((Number*)nptr)->value;
            /*
            if (max_len < 0)
                throw use_error("maxItems: is negative");
            */
            if (max_len != ((Number*)nptr)->value)
                throw use_error("maxItems: not an integer");
        }
        nptr = obj->get("minItems");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("minItems: not a number");
            min_len = ((Number*)nptr)->value;
            /*
            if (min_len < 0)
                throw use_error("minItems: is negative");
            */
            if (min_len != ((Number*)nptr)->value)
                throw use_error("minItems: not an integer");
            if (max_len < min_len)
                throw use_error("minItems greater than maxItems");
        }
        nptr = obj->get("uniqueItems");
        if (nptr != nullptr) {
            if (nptr->type() != Type::BOOLEAN)
                throw use_error("uniqueItems: not a bool");
            unique_items = (Bool*)nptr == &Bool::T;
        }
    } else if (s_type == "object") {
        nptr = obj->get("properties");
        if (nptr != nullptr) {
            if (nptr->type() != Type::OBJECT)
                throw use_error("properties: not an object");
            Object* src = (Object*)nptr;
            props = new Object();
            for (auto kv : src->map) {
                Schema* sp = new Schema(kv.second);
                props->set(*kv.first, sp);
            }
            props->refcnt++;
        }
        nptr = obj->get("patternProperties");
        if (nptr != nullptr) {
            if (nptr->type() != Type::OBJECT)
                throw use_error("patternProperties: not an object");
            Object* src = (Object*)nptr;
            pat_props = new Object();
            for (auto kv : src->map) {
                Schema* sp = new Schema(kv.second);
                pat_props->set(*kv.first, sp);
                // TODO *kv.first should be regex
            }
            pat_props->refcnt++;
        }
        nptr = obj->get("additionalProperties");
        if (nptr != nullptr) {
            if (nptr->type() == Type::OBJECT)
                add_props = new Schema(nptr);
            else if (nptr->type() == Type::BOOLEAN)
                add_props_bool = (Bool*)nptr == &Bool::T;
            else
                throw use_error("additionalProperties: not an object or bool");
        }
        nptr = obj->get("maxProperties");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("maxProperties: not a number");
            max_len = ((Number*)nptr)->value;
            /*
            if (max_len < 0)
                throw use_error("maxProperties: is negative");
            */
            if (max_len != ((Number*)nptr)->value)
                throw use_error("maxProperties: not an integer");
        }
        nptr = obj->get("minProperties");
        if (nptr != nullptr) {
            if (nptr->type() != Type::NUMBER)
                throw use_error("minProperties: not a number");
            min_len = ((Number*)nptr)->value;
            /*
            if (min_len < 0)
                throw use_error("minProperties: is negative");
            */
            if (min_len != ((Number*)nptr)->value)
                throw use_error("minProperties: not an integer");
            if (max_len < min_len)
                throw use_error("minProperties greater than maxProperties");
        }
        nptr = obj->get("required");
        if (nptr != nullptr) {
            if (nptr->type() != Type::ARRAY)
                throw use_error("required: not an array");
            required = (Array*)nptr;
            required->refcnt++;
            for (Node* n : required->list) {
                if (n->type() != Type::STRING)
                    throw use_error("required: not an array of strings");
            }
        }
        // TODO dependencies
    } else if (s_type != "boolean" && s_type != "null")
        throw use_error("type: illegal value");
}

Json::Schema::~Schema() {
    if (s_enum != nullptr)
        s_enum->unref();
    for (Schema* sp : allof)
        delete sp;
    for (Schema* sp : anyof)
        delete sp;
    for (Schema* sp : oneof)
        delete sp;
    if (s_not != nullptr)
        delete s_not;
    if (pattern != nullptr)
        delete pattern;
    if (item != nullptr)
        delete item;
    for (Schema* sp : items)
        delete sp;
    if (add_items != nullptr)
        delete add_items;
    if (props != nullptr)
        props->unref();
    if (pat_props != nullptr)
        pat_props->unref();
    if (add_props != nullptr)
        delete add_props;
    if (required != nullptr)
        required->unref();
    if (defs != nullptr)
        defs->unref();
    if (deflt != nullptr)
        deflt->unref();
    // TODO the rest
}

void Json::Node::validate(const Schema& schema, vector<const Node*>& path) const {
    if (schema.s_enum != nullptr) {
        bool found = false;
        for (const Node* n : schema.s_enum->list) {
            if (*this == *n) {
                found = true;
                break;
            }
        }
        if (!found)
            throw invalid("value not in enum");
    }
    for (Schema* sp : schema.allof)
        validate(*sp, path);
    if (schema.anyof.size() > 0) {
        bool ok = false;
        for (Schema* sp : schema.anyof) {
            try {
                validate(*sp, path);
                ok = true;
                break;
            } catch (invalid& ex) {
                path.pop_back();  // TODO ???
            }
        }
        if (!ok)
            throw invalid("all anyOf validations failed");
    }
    // TODO oneof, not, definitions
}

void Json::Array::validate(const Schema& schema, vector<const Node*>& path) const {
    path.push_back(this);
    if (schema.s_type != "array")
        throw invalid("type mismatch");
    if (list.size() < schema.min_len)
        throw invalid("array length below minItems");
    if (list.size() > schema.max_len)
        throw invalid("array length above maxItems");
    // TODO uniqueItems
    if (schema.add_items == nullptr) {
        if (!schema.add_items_bool) {
            if (schema.item == nullptr && list.size() > schema.items.size())
                throw invalid("array too long");
        }
    }
    if (schema.item != nullptr) {
        for (Node* it : list) {
            it->validate(*schema.item, path);
        }
    } else {
        for (unsigned i = 0; i < list.size(); i++) {
            Schema* sub = i < schema.items.size() ? schema.items[i] : schema.add_items;
            list[i]->validate(*sub, path);
        }
    }
    path.pop_back();
}

void Json::Object::validate(const Schema& schema, vector<const Node*>& path) const {
    path.push_back(this);
    if (schema.s_type != "object")
        throw invalid("type mismatch");
    if (map.size() < schema.min_len)
        throw invalid("number of properties below minimum");
    if (map.size() > schema.max_len)
        throw invalid("number of properties above maximum");
    if (schema.required != nullptr) {
        for (Node* prop : schema.required->list) {
            if (get(((String*)prop)->value) == nullptr)
                throw invalid("required property missing");
        }
    }
    if (schema.props != nullptr && schema.add_props == nullptr && !schema.add_props_bool) {
        for (auto kv : map) {
            const string* k = kv.first;
            if (schema.props->map.find(k) == schema.props->map.end())
                throw invalid(*k + ": not valid in schema");
            // TODO: find in patpr
        }
    }
    for (auto kv : map) {
        const string* k = kv.first;
        if (schema.props == nullptr ||
                schema.props->map.find(k) == schema.props->map.end()) {
            if (schema.add_props == nullptr && schema.add_items_bool)
                continue;  // allowed not to validate
            if (schema.add_props != nullptr)
                kv.second->validate(*schema.add_props, path);
            // TODO find in patpr
        } else {
            Schema* sub = (Schema*)schema.props->map[k];
            kv.second->validate(*sub, path);
        }
    }
    // TODO dependencies
    path.pop_back();
}

void Json::Number::validate(const Schema& schema, vector<const Node*>& path) const {
    path.push_back(this);
    if (schema.s_type != "number")
        throw invalid("type mismatch");
    if (value < schema.min_num)
        throw invalid("number below minimum");
    if (value == schema.min_num && schema.min_exc)
        throw invalid("number not above minimum");
    if (value > schema.max_num)
        throw invalid("number above maximum");
    if (value == schema.max_num && schema.max_exc)
        throw invalid("number not below maximum");
    if (schema.mult_of != 0) {
        long double quot = value / schema.mult_of;
        if (quot != (long long)quot)
            throw invalid("number not a multiple of");
    }
    path.pop_back();
}

/**
 * Returns number of utf-8 characters in std::string.
 * Possible encoding errors are ignored.
 */
static unsigned u8size(const string& s) {
    unsigned off = 0, count = 0;
    unsigned len = s.length();
    while (off < len) {
        unsigned char c = s[off];
        count++;
        if ((c & 0x80) == 0)
            off += 1;
        else if ((c & 0b11100000) == 0b11000000)
            off += 2;
        else if ((c & 0b11110000) == 0b11100000)
            off += 3;
        else if ((c & 0b11111000) == 0b11110000)
            off += 4;
    }
    return count;
}

void Json::String::validate(const Schema& schema, vector<const Node*>& path) const {
    path.push_back(this);
    if (schema.s_type != "string")
        throw invalid("type mismatch");
    if (u8size(value) < schema.min_len)
        throw invalid("string length below minLength");
    if (u8size(value) > schema.max_len)
        throw invalid("string length above maxLength");
    /*
    const String* pattern = schema->getstr("pattern", false);
    if (pattern != nullptr) {
        try {
            regex rex(pattern->value, regex_constants::ECMAScript);
            if (!regex_search(value, rex))
                throw invalid("pattern mismatch");
        } catch (regex_error& ex) {
            cout << ex.what() << ' ' << ex.code() << '\n';
            cout << "pattern :'" << pattern->value << "'\n";
            throw use_error(ex.what());
        }
    } */
    // TODO pattern
    path.pop_back();
}

bool Json::to_schema(string* reason) {
    try {
        if (root->is_schema())
            return true;
        Schema* sp = new Schema(root);
        root->unref();
        (root = sp)->refcnt++;
        return true;
    } catch (use_error& ex) {
        if (reason != nullptr)
            *reason = ex.what();
    }
    return false;
}

bool Json::valid(Json& schema, string* reason) {
    if (!schema.root->is_schema()) {
        if (!schema.to_schema(reason))
            return false;
    }
    vector<const Node*> path;
    try {
        ((Node*)root)->validate(*(Schema*)schema.root, path);
        root->validate(*(Schema*)schema.root, path);
    } catch (invalid& ex) {
        string pref = "";
        for (unsigned i = 1; i < path.size(); i++) {
            const Node* super = path[i-1];
            const Node* curr = path[i];
            if (super->type() == Type::ARRAY) {
                auto list = ((Array*)super)->list;
                auto it = find(list.begin(), list.end(), curr);
                pref += "[" + to_string(it - list.begin()) + "]";
            } else if (super->type() == Type::OBJECT) {
                auto map = ((Object*)super)->map;
                for (auto kv : map) {
                    if (kv.second == curr) {
                        pref += "." + *kv.first;
                        break;
                    }
                }
            }
        }
        if (reason != nullptr)
            *reason = pref + ": " + ex.msg;
        return false;
    }
    return true;
}

#endif


