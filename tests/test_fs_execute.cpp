
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/execute.h>

#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[filesystem-execute] Test command execution via filesystem::detail::execute",
              "[fs-execute-test-command-execution-filesystem-detail-execute]")
    {
        std::string output;
        std::string command;
        
        // command = "curl -sS 'https://medium.com/@hopro/homeless-tips-time-space-data-power-ccbb6338c59f#.5obyn0apb' | grep -i obvinit";
        // output = filesystem::detail::execute(command.c_str());
        // CHECK(output.find("obvInit") != std::string::npos);
        
        command = "ps aux";
        output = filesystem::detail::execute(command.c_str());
        CHECK(output.find("ps aux") != std::string::npos);
    }
    
    
} /// namespace (anon.)