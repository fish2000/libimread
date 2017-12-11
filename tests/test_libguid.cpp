
#define CATCH_CONFIG_FAST_COMPILE

#include <libimread/errors.hh>
#include <guid.h>
#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[libguid] Run bundled tests",
              "[libguid-run-bundled-tests]")
    {
        
        GuidGenerator generator = GuidGenerator();
        
        Guid r1 = generator.newGuid();
        Guid r2 = generator.newGuid();
        Guid r3 = generator.newGuid();
        
        // WTF("YO DOGG:", "generator.newGuid() output is as follows:", r1, r2, r3);
        
        CHECK(r1 != r2);
        CHECK(r1 != r3);
        CHECK(r2 != r3);
        
        // {
        //   outStream << "FAIL - not all random guids are different" << endl;
        //   return 1;
        // }
        
        Guid s1("7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        Guid s2("16d1bd03-09a5-47d3-944b-5e326fd52d27");
        Guid s3("fdaba646-e07e-49de-9529-4499a5580c75");
        Guid s4("7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        
        CHECK(r1 != s1);
        CHECK(r2 != s2);
        CHECK(r3 != s3);
        
        CHECK(s1 != s2);
        CHECK(s1 != s3);
        CHECK(s2 != s3);
        
        CHECK(s1 == s4);
        CHECK(s4 == s1);
        
        CHECK(s1.str() == "7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        CHECK(s2.str() == "16d1bd03-09a5-47d3-944b-5e326fd52d27");
        CHECK(s3.str() == "fdaba646-e07e-49de-9529-4499a5580c75");
        CHECK(s4.str() == "7bcd757f-5b10-4f9b-af69-1a1f226f3b3e");
        
    }
};
