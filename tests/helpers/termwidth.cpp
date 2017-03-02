#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>

using winsize_t = struct winsize;

namespace im {
    namespace test {
        int termwidth() {
            winsize_t winsize;
            ::ioctl(STDOUT_FILENO, TIOCGWINSZ, &winsize);
            return winsize.ws_row;
        }
    }
}