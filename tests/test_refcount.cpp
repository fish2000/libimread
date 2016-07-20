
#include <libimread/libimread.hpp>
#include <libimread/ext/memory/refcount.hh>
#include <libimread/errors.hh>

#include "include/catch.hpp"
#include "helpers/ByteHeap.hh"

namespace {
    
    using memory::RefCount;
    using im::test::ByteHeap;
    using im::byte;
    
    struct Trivial {
        volatile int integer;
        Trivial()
            :integer(0)
            {
                // WTF("Trivial::Trivial() copy constructor");
            }
        virtual ~Trivial() {
            // WTF("Trivial::~Trivial() destructor");
        }
    };
    
    TEST_CASE("[refcount] Test RefCount basic operations",
              "[refcount-basic-operations]")
    {
        auto count = RefCount<Trivial>(new Trivial());
        CHECK(count.retainCount() == 1);
        
        auto cc(count);
        CHECK(count.retainCount() == 2);
        
        cc.retain();
        CHECK(count.retainCount() == 3);
        
        cc.release();
        CHECK(count.retainCount() == 2);
        
    }
    
    /*
    TEST_CASE("[refcount] Test RefCount with ByteHeap",
              "[refcount-with-byteheap]")
    {
        constexpr int SIZE = 128;
        
        {
            auto heap = RefCount<ByteHeap<byte>>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<ByteHeap<byte>>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<ByteHeap<byte>>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<ByteHeap<byte>>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
    }
    */
    
}

