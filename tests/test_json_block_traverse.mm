
#import <Foundation/Foundation.h>
#import <BlockClosure.h>

#include <libimread/libimread.hpp>
#include <libimread/ext/JSON/json11.h>
#include <libimread/errors.hh>

#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[json-block-traverse] Traverse JSON tree with a block literal via MABlockClosure",
              "[json-block-traverse-with-block-literal]")
    {
        using Node = Json::JSONNode;
        using fptr_t = std::add_pointer_t<void(const Node*)>;
        Json dict;
        dict["one"] = "one.";
        dict["two"] = "two.";
        dict["three"] = { 435, 345987, 238746, 21 };
        
        id block = ^(void* node) {
            WTF("JSONNode found: ", ((Node *)node)->typestr());
        };
        [block retain];
        
        dict.traverse((fptr_t)BlockFptr(block));
        [block release];
    }
    
}

