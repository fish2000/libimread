
#include <libimread/errors.hh>
#include <guid.h>
#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[libguid] Run bundled tests",
              "[libguid-run-bundled-tests]")
    {
        
        GuidGenerator generator = GuidGenerator();
        
        auto r1 = generator.newGuid();
        auto r2 = generator.newGuid();
        auto r3 = generator.newGuid();
        
        // WTF("YO DOGG: generator.newGuid output as follows:", r1, r2, r3);
        
        Guid s1("7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        Guid s2("16d1bd03-09a5-47d3-944b-5e326fd52d27");
        Guid s3("fdaba646-e07e-49de-9529-4499a5580c75");
        Guid s4("7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        
        CHECK(r1 != r2);
        CHECK(r1 != r3);
        CHECK(r2 != r3);
        
        // {
        //   outStream << "FAIL - not all random guids are different" << endl;
        //   return 1;
        // }
        
        CHECK(s1 != s2);
        CHECK(s1 == s4);
        CHECK(s1.str() == "7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        CHECK(s2.str() == "16d1bd03-09a5-47d3-944b-5e326fd52d27");
        CHECK(s3.str() == "fdaba646-e07e-49de-9529-4499a5580c75");
        
    }
};
