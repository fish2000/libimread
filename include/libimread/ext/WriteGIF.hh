
#ifndef LIBIMREAD_EXT_WRITEGIF_HH_
#define LIBIMREAD_EXT_WRITEGIF_HH_

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <vector>
#include <memory>

#include <libimread/libimread.hpp>

using im::byte;

namespace gif {
    struct GIF;
    GIF *newGIF(int delay = 3);
    void dispose(GIF *gif);
    void addFrame(GIF *gif, int W, int H, unsigned char *rgbImage, int delay = -1);
    std::vector<byte> write(GIF *gif);
}

#endif /// LIBIMREAD_EXT_WRITEGIF_HH_