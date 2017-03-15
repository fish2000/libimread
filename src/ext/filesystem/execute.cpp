
#include <array>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/execute.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>

namespace filesystem {
    
    namespace detail {
        
        std::string execute(char const* command, char const* workingdir) {
            std::array<char, EXECUTION_BUFFER_SIZE> buffer;
            std::string result;
            std::unique_ptr<FILE, decltype(::pclose)&> pipe(::popen(command, "r"), ::pclose);
            switchdir s(workingdir ? path(workingdir) : path::tmp());
            if (!pipe.get()) { return NULL_STR; }
            while (!std::feof(pipe.get())) {
                if (std::fgets(buffer.data(), EXECUTION_BUFFER_SIZE, pipe.get()) != nullptr) {
                    result += buffer.data();
                }
            }
            return result;
        }
        
    }
    
}