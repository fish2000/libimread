
#define NO_IMPORT_ARRAY

#include <libimread/IO/ppm.hh>
#include <libimread/ext/fmemopen.hh>


#define SWAP_ENDIAN16(little_endian, value) \
    if (little_endian) { (value) = (((value) & 0xff)<<8)|(((value) & 0xff00)>>8); }

namespace im {
    
    namespace {
        
        inline int is_little_endian() {
            int value = 1;
            return ((char *) &value)[0] == 1;
        }
        
        template <typename T = uint8_t>
        inline T *at(Image &im, int x, int y, int z) {
            return &im.rowp_as<T>(0)[x*im.stride(0) +
                                      y*im.stride(1) +
                                      z*im.stride(2)];
        }
        
    }
    
    std::unique_ptr<Image> PPMFormat::read(byte_source *src,
                                           ImageFactory *factory,
                                           const options_map &opts) {
        /// YO DOGG
        std::vector<byte> all = src->full_data();
        memory::buffer membuf = memory::source(&all[0], all.size());
        
        int width, height, maxval;
        char header[256];
        _ASSERT(fscanf(membuf.get(), "%255s", header) == 1, "Could not read PPM header\n");
        _ASSERT(fscanf(membuf.get(), "%d %d\n", &width, &height) == 2, "Could not read PPM width and height\n");
        _ASSERT(fscanf(membuf.get(), "%d", &maxval) == 1, "Could not read PPM max value\n");
        _ASSERT(fgetc(membuf.get()) != EOF, "Could not read char from PPM\n");
        
        int bit_depth = 0;
        if (maxval == 255) { bit_depth = 8; }
        else if (maxval == 65535) { bit_depth = 16; }
        else { _ASSERT(false, "Invalid bit depth in PPM\n"); }
        
        _ASSERT(strcmp(header, "P6") == 0 || strcmp(header, "p6") == 0, "Input is not binary PPM\n");
        
        const int channels = 3;
        std::unique_ptr<Image> im = factory->create(bit_depth, height, width, channels);
        const int full_size = width*height*3;
        
        // convert the data to T
        if (bit_depth == 8) {
            uint8_t *data = new uint8_t[full_size];
            _ASSERT(fread(static_cast<void*>(data),
                          sizeof(uint8_t), full_size, membuf.get()) == static_cast<size_t>(full_size),
                    "Could not read PPM 8-bit data\n");
            
            uint8_t *im_data = im->rowp_as<uint8_t>(0);
            for (int y = 0; y < height; y++) {
                uint8_t *row = static_cast<uint8_t *>(&data[(y*width)*3]);
                for (int x = 0; x < width; x++) {
                    pix::convert(*row++, im_data[(0*height+y)*width+x]);
                    pix::convert(*row++, im_data[(1*height+y)*width+x]);
                    pix::convert(*row++, im_data[(2*height+y)*width+x]);
                }
            }
            delete[] data;
        } else if (bit_depth == 16) {
            int little_endian = is_little_endian();
            uint16_t *data = new uint16_t[full_size];
            _ASSERT(fread(static_cast<void*>(data),
                          sizeof(uint16_t), full_size, membuf.get()) == static_cast<size_t>(full_size),
                "Could not read PPM 16-bit data\n");
            
            uint16_t *im_data = im->rowp_as<uint16_t>(0);
            for (int y = 0; y < height; y++) {
                uint16_t *row = static_cast<uint16_t *>(&data[(y*width)*3]);
                for (int x = 0; x < width; x++) {
                    uint16_t value;
                    value = *row++; SWAP_ENDIAN16(little_endian, value); pix::convert(value, im_data[(0*height+y)*width+x]);
                    value = *row++; SWAP_ENDIAN16(little_endian, value); pix::convert(value, im_data[(1*height+y)*width+x]);
                    value = *row++; SWAP_ENDIAN16(little_endian, value); pix::convert(value, im_data[(2*height+y)*width+x]);
                }
            }
            delete[] data;
        }
        
        return im;
    }
    
    void PPMFormat::write(Image &input, byte_sink *output,
                          const options_map &opts) {
        /// YO DOGG
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2);
        const int bit_depth = input.nbits();
        const int full_size = width*height*3;
        
        /// write header
        output->writef("P6\n%d %d\n%d\n",
            width, height, (1<<bit_depth)-1);
        
        if (bit_depth == 8) {
            uint8_t *data = new uint8_t[full_size];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    uint8_t *p = static_cast<uint8_t *>(&data[(y*width+x)*3]);
                    for (int c = 0; c < channels; c++) {
                        pix::convert(at(input, x, y, c)[0], p[c]);
                    }
                }
            }
            _ASSERT(output->write(data, full_size) == static_cast<size_t>(full_size),
                "Could not write PPM 8-bit data\n");
            delete[] data;
        } else if (bit_depth == 16) {
            int little_endian = is_little_endian();
            uint16_t *data = new uint16_t[full_size];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    uint16_t *p = static_cast<uint16_t *>(&data[(y*width+x)*3]);
                    for (int c = 0; c < channels; c++) {
                        uint16_t value;
                        pix::convert(at<uint16_t>(input, x, y, c)[0], value);
                        SWAP_ENDIAN16(little_endian, value);
                        p[c] = value;
                    }
                }
            }
            _ASSERT(output->write(data, full_size) == static_cast<size_t>(full_size),
                "Could not write PPM 16-bit data\n");
            delete[] data;
        }
        
        output->flush();
    }
}

