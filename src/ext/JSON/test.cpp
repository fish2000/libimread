/*
 * This file is part of json11 project (https://github.com/borisgontar/json11).
 *
 * Copyright (c) 2013 Boris Gontar.
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

#include <libimread/ext/JSON/json11.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace std;

void expect(string test, ostringstream& out, string wanted) {
    string result = out.str().substr(0, out.tellp());
    //cout << result << '\n';
    if (result != wanted)
        cout << test << " failed: actual/expected:\n" << result << '\n' << wanted << '\n';
    else
        cout << test << " ok\n";
    out.seekp(0);
}

void proc(Json j, Json a) {
    j["arr"] = a;
    a[2] = "bye";
}

void test() {
    ostringstream out;
    //
    // test0: scalars
    Json null, inum = 3, lnum = 68719476736L, fnum = 3.14f, dnum = 2.718281828459;
    Json ldnum = 2.71828182845904523L;
    Json t = true, f = false, str = "hello";
    out << null << ' ' << inum << ' ' << lnum << ' ' << fnum << ' ';
    out << dnum << ' ';
    out << ldnum << ' ';
    out << t << ' ' << f << ' ' << str;
    expect("test0", out, "null 3 68719476736 3.14 2.718281828459 "
            "2.71828182845904523 true false \"hello\"");
    //
    // test1: casts
    if (t && !f) {
        int ival = inum;
        long long lval = lnum;
        float fval = fnum;
        double dval = dnum;
        long double ldval = ldnum;
        out << ival << ' ' << lval << ' ' << setprecision(FLT_DIG) << fval << ' ';
        out << setprecision(DBL_DIG) << dval << ' ';
        out << setprecision(LDBL_DIG) << ldval << ' ' << (string)str;
        expect("test1", out, "3 68719476736 3.14 2.718281828459 2.71828182845904523 hello");
    } else
        cout << "test1 bad bools\n";
    //
    // test2: arrays
    Json arr = Json::array();
    arr << null << inum << lnum << fnum << dnum << ldnum << t << f << str;
    str = "bye";  // should not affect arr
    out << arr;
    expect("test2", out, "[null,3,68719476736,3.14,2.718281828459,"
            "2.71828182845904523,true,false,\"hello\"]");
    //
    // test3: objects
    Json obj;
    obj["i"] = 2;
    obj.set("l", 4294967296).set("ll", lnum);
    obj.set("f", 1.23).set("d", 0.98).set("ld", ldnum);
    obj.set("true", true).set("false", false).set("null", Json::null);
    lnum = 0; ldnum = 0;  // should not affect obj
    out << obj;
    expect("test3", out, R"/({"i":2,"l":4294967296,"ll":68719476736,"f":1.23,)/"
            R"/("d":0.98,"ld":2.71828182845904523,"true":true,"false":false,"null":null})/");
    //
    // test4: parsing
    string js = obj.format();
    istringstream istr(js);
    Json obj2(istr);
    if (obj != obj2)
        cout << "test4 failed\n";
    else
        cout << "test4 ok\n";
    //
    // test5: subscript rhs
    Json obj3 = Json::parse(R"/( {"a":[{"x":"pro\"s\tand con\"s"}], "b":[], "c": {"z": [1,2,3]} } )/");
    string qq = obj3["a"][0]["x"];
    out << qq << ' ' << obj3["b"] << ' ' << obj3["c"]["z"][1] << ' ';
    out << arr[0] << ' ' << arr[4] << ' ' << obj["d"];
    expect("test5", out, "pro\"s\tand con\"s [] 2 null 2.718281828459 0.98");
    //
    // test6: subscript lhs
    Json a123 {1, 2, "three"};
    a123 << Json::parse(R"/({"u":4, "v":5, "w":6})/");
    a123[3]["v"] = "five";
    out << a123 << ' ';
    Json three = a123[2];
    a123[2] = 3;
    out << three << ' ';
    Json objf = obj["f"];
    obj["f"] = a123[3]["w"];
    out << objf << ' ' << obj["f"];
    expect("test6", out, "[1,2,\"three\",{\"u\":4,\"v\":\"five\",\"w\":6}] \"three\" 1.23 6");
    //
    // test7: exceptions
    try {
        int x = three;
        cout << "test7 failed: " << x;
    } catch (std::exception& ex) {
        if (string(ex.what()) != "std::bad_cast")
            cout << "test7: catched: " << ex.what() << ", expected std::bad_cast\n";
    }
    try {
        Json js1 { 1, 2 }, js2;
        js2 << js1;
        js1 << js2;
        cout << "test7 failed: cyclic dependency\n";
    } catch (Json::use_error& ex) {
        if (string(ex.what()) != "cyclic dependency")
            cout << "text7 failed: " << ex.what() << '\n';
    }
    //
    // test8: schema
    ifstream fs1("products.json");
    Json prods(fs1);
    fs1.close();
    ifstream fs2("products-schema.json");
    Json sch(fs2);
    fs2.close();
    string reason;
    if (!sch.to_schema(&reason))
        cout << "test8 schema: " << reason << '\n';
    else if (prods.valid(sch, &reason))
        cout << "test8 ok\n";
    else
        cout << "test8: " << reason << '\n';
}

int main(int argc, char** argv) {
    try {
        test();
    } catch (Json::use_error& ex) {
        cout << "use_error: " << ex.what() << '\n';
    } catch (Json::parse_error& ex) {
        cout << "parse_error: " << ex.what() << '\n';
        cout << "line: " << ex.line << ", col: " << ex.col << '\n';
    } catch (std::exception& ex) {
        cout << "exception: " << ex.what() << '\n';
    }
#ifdef TEST
	Json::test();
#endif
}
