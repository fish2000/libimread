
#include <libimread/libimread.hpp>
#include <libimread/ext/memory/refcount.hh>
#include <libimread/errors.hh>

#include "include/catch.hpp"
#include "helpers/HeapAllocation.hh"

namespace {
    
    using memory::RefCount;
    using im::test::Bytes;
    using im::test::Chars;
    using im::byte;
    
    struct Trivial {
        volatile int integer;
        Trivial(int value = 0)
            :integer(value)
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
    
    TEST_CASE("[refcount] Test RefCount with HeapAllocation<byte>",
              "[refcount-with-heapallocation-byte]")
    {
        constexpr int SIZE = 128;
        
        {
            auto heap = RefCount<Bytes>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<Bytes>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        memory::garbageday();
        {
            auto heap = RefCount<Bytes>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<Bytes>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
    }
    
    TEST_CASE("[refcount] Test RefCount with HeapAllocation<char>",
              "[refcount-with-heapallocation-char]")
    {
        constexpr int SIZE = 128;
        
        {
            auto heap = RefCount<Chars>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<Chars>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        memory::garbageday();
        {
            auto heap = RefCount<Chars>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
        {
            auto heap = RefCount<Chars>::MakeRef(SIZE);
            auto s = heap->size();
            REQUIRE(s == SIZE);
        }
    }
    
}

