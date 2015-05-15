
#define NO_IMPORT_ARRAY

#include "numpy.hh"

namespace im {
    
    /*
    void NumpyImage::finalize() {
        if (PyArray_TYPE(array) == NPY_BOOL) {
            // We need to expand this
            const int h = PyArray_DIM(array, 0);
            const int w = PyArray_DIM(array, 1);
            std::vector<byte> buf;
            buf.resize(w);
            for (int y = 0; y != h; ++y) {
                byte *data = static_cast<byte*>(PyArray_GETPTR1(array, y));
                for (int x = 0; x != ((w/8)+bool(w%8)); ++x) {
                    const byte v = data[x];
                    for (int b = 0; b != 8 && (x*8+b < w); ++b) {
                        buf[x*8+b] = bool(v & (1 << (7-b)));
                    }
                }
                std::memcpy(data, &buf[0], w);
            }
        }
    }
    */

}