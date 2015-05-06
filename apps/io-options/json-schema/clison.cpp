/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

// Version 0.6.5, 2013-11-07

#include "json11.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace std;

Json meta;
Json schema;
Json sprops;
const char* typetag[] = { "null", "bool", "number", "string", "array", "object" };

Json follow(Json& top, vector<string>& path, string* full) {
    Json curr = top;
    for (string key : path) {
        if (curr.type() == Json::ARRAY) {
            curr = curr[stoi(key)];
            if (full != nullptr)
                *full += "[" + key + "]";
        } else if (curr.type() == Json::OBJECT) {
            curr = curr[key];
            if (full != nullptr)
                *full += "." + key;
        } else
            throw Json::use_error("");
    }
    return curr;
}

void cli(Json& top, Json js, vector<string>& path) {
start:
    Json::Type type = js.type();
    string fullpath;
    follow(top, path, &fullpath);
    cout << (fullpath == "" ? "at top" : fullpath) << ": " << typetag[type] << endl;
    if (type == Json::Type::OBJECT) {
        int n = 0;
        for (string key : js.keys()) {
            Json prop = js.get(key);
            cout << setw(3) << n++ << ". " << key;
            if (prop.type() == Json::ARRAY)
                cout << " [" << prop.size() << "]";
            else if (prop.type() == Json::OBJECT)
                cout << " {" << prop.size() << "}";
            else
                cout << ": " << prop;
            cout << endl;
        }
    } else if (type == Json::Type::ARRAY) {
        int n = 0;
        for (unsigned i = 0; i < js.size(); i++) {
            cout << setw(3) << n++ << ". ";
            Json elem = js[i];
            Json::Type eltype = elem.type();
            if (eltype == Json::ARRAY)
                cout << typetag[eltype] << " [" << elem.size() << "]";
            else if (eltype == Json::OBJECT)
                cout << typetag[eltype] << " {" << elem.size() << "}";
            else
                cout << elem;
            cout << endl;
        }
    } else
        cout << js << endl;
    char line[1024];
    do {
        cout << "> ";
        cin.getline(line, sizeof(line));
        if (cin.eof()) {
            cout << '\n';
            exit(0);
        }
        char* p = line;
        while (isspace(*p))
            p++;
        if (*p == 0)
            continue;
        if (*p == 'h' || *p == '?') {
            cout << "enter a number to select an object, q to go back\n";
            cout << ".             : list current object\n";
            cout << "p [file.json] : print out current object [into file]\n";
#ifdef WITH_SCHEMA
            cout << "s file.json   : load file as a json schema\n";
#endif
            cout << "= text        : replace current object by parsed text\n";
            continue;
        }
        if (*p == '.')
            goto start;
        if (*p == 'q') {
            if (!path.empty())
                path.pop_back();
            return;
        }
        if (*p == 'p') {
            while (isspace(*++p));
            if (*p == 0) {
                cout << js << '\n';
                continue;
            }
            ofstream out(p);
            if (!out) {
                cout << "cannot write to '" << p << "'\n";
                continue;
            }
            out << js << '\n';
            if (out.bad())
                cout << "i/o error occured while writing\n";
            out.close();
            continue;
        }
#ifdef WITH_SCHEMA
        if (*p == 's') {
            while (isspace(*++p));
            if (*p == 0) {
                cout << "file name expected\n";
                continue;
            }
            ifstream ifs(p);
            if (!ifs) {
                cout << "cannot read " << p << '\n';
                continue;
            }
            try {
                ifs >> schema;
            } catch (std::exception& ex) {
                cout << "exception: " << ex.what() << '\n';
            }
            string reason;
            if (!js.valid(schema, &reason))
                cout << reason << '\n';
            continue;
        }
#endif
        if (*p == '=') {
            try {
                js = Json::parse(p + 1);
                if (path.empty()) {
                    top = js;
                    continue;
                }
                vector<string> ppath(path.begin(), path.end()-1);
                Json parent = follow(top, ppath, nullptr);
                if (parent.type() == Json::Type::ARRAY)
                    parent.replace(stoi(path.back()), js);
                else
                    parent.set(path.back(), js);
            }catch (Json::use_error& ex) {
                cout << "use_error: " << ex.what() << '\n';
            } catch (Json::parse_error& ex) {
                cout << "parse_error: " << ex.what() << '\n';
                cout << "line: " << ex.line << ", col: " << ex.col << '\n';
            }
            continue;
        }
        int n = -1;
        while (isdigit(*p)) {
            if (n < 0)
                n = 0;
            n = 10 * n + (*p++ - '0');
        }
        if (n < 0) {
            cout << "?  (type 'h' for help)\n";
            continue;
        }
        if ((type == Json::OBJECT || type == Json::ARRAY) && n >= (int)js.size()) {
            cout << "out of range\n";
            continue;
        }
        Json next;
        string name;
        if (type == Json::OBJECT) {
            name = js.keys()[n];
            next = js[name];
        } else if (type == Json::ARRAY) {
            name = to_string(n);
            next = js[n];
        }
        if (next.type() != Json::JSNULL) {
            path.push_back(name);
            cli(top, next, path);
            goto start;
        }
    } while (true);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: clison filename.json\n";
        return 1;
    }
    try {
#ifdef WITH_SCHEMA
        ifstream schfs("schema.json");
        if (!schfs)
            cout << "note: schema.json not found\n";
        else {
            //meta = Json(schfs);
            schfs >> meta;
            string schema_url = "http://json-schema.org/draft-04/schema#";
            if (meta.type() != Json::OBJECT || meta["$schema"] != schema_url) {
                cout << "schema.json is not a http://json-schema.org/draft-04/schema\n";
                meta = Json::null;
            }
            schfs.close();
            sprops = meta["properties"];
        }
#endif
        ifstream fs(argv[1]);
        Json js(fs);
        fs.close();
        Json::indent = 2;
        vector<string> path;
        cli(js, js, path);
    } catch (Json::use_error& ex) {
        cout << "use_error: " << ex.what() << '\n';
    } catch (Json::parse_error& ex) {
        cout << "parse_error: " << ex.what() << '\n';
        cout << "line: " << ex.line << ", col: " << ex.col << '\n';
    } catch (std::exception& ex) {
        cout << "exception: " << ex.what() << '\n';
    }
#ifdef TEST
    meta = Json::null;
    schema = Json::null;
    Json::test();
#endif
}
