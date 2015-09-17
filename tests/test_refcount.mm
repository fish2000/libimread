
#include <libimread/libimread.hpp>
#include <libimread/ext/memory/refcount.hh>
#include <libimread/errors.hh>

#include "include/catch.hpp"
#include "helpers/ByteHeap.hh"

namespace {
    
    using memory::RefCount;
    using im::test::ByteHeap;
    
    struct Trivial {
        volatile int integer;
        Trivial()
            :integer(0)
            {
                WTF("Trivial::Trivial() copy constructor");
            }
        virtual ~Trivial() {
            WTF("Trivial::~Trivial() destructor");
        }
    };
    
    TEST_CASE("[refcount] Test RefCount basic operations",
              "[refcount-basic-operations]")
    {
        auto count = RefCount<Trivial>(new Trivial());
        REQUIRE(count.retainCount() == 1);
        
    }
    
    // TEST_CASE("[refcount] Test RefCount with ByteHeap",
    //           "[refcount-with-byteheap]")
    // {}
    
}

