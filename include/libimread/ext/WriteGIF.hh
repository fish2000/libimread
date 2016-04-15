
#ifndef LIBIMREAD_EXT_WRITEGIF_HH_
#define LIBIMREAD_EXT_WRITEGIF_HH_

#include <vector>
#include <libimread/libimread.hpp>

using im::byte;

namespace gif {
    struct GIF;
    using bytevec_t = std::vector<byte>;
    GIF* newGIF(int delay = 3);
    void dispose(GIF* gif);
    void addFrame(GIF* gif, int W, int H, unsigned char* rgbImage, int delay = -1);
    bytevec_t write(GIF* gif);
}

#endif /// LIBIMREAD_EXT_WRITEGIF_HH_