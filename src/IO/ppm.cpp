/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#define NO_IMPORT_ARRAY

#include <sstream>
#include <cstdio>
#include <cstring>

#include <iod/json.hh>
#include <libimread/IO/ppm.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>
#include <libimread/ext/memory/fmemopen.hh>
#include <libimread/base.hh>

#define SWAP_ENDIAN16(value) \
    (value) = (((value) & 0xff)<<8)|(((value) & 0xff00)>>8)

namespace im {
    
    DECLARE_FORMAT_OPTIONS(PPMFormat);
    
    std::unique_ptr<Image> PPMFormat::read(byte_source* src,
                                           ImageFactory* factory,
                                           Options const& opts) {
        /// YO DOGG
        bytevec_t all = src->full_data();
        memory::buffer membuf = memory::source(&all[0], all.size());
        
        int width, height, maxval;
        char header[256];
        
        imread_assert(std::fscanf(membuf.get(), "%255s", header) == 1,
                      "Could not read PPM header\n");
        
        imread_assert(std::fscanf(membuf.get(), "%d %d\n", &width, &height) == 2,
                      "Could not read PPM width and height\n");
        
        imread_assert(std::fscanf(membuf.get(), "%d", &maxval) == 1,
                      "Could not read PPM max value\n");
        
        imread_assert(std::fgetc(membuf.get()) != EOF,
                      "Could not read char from PPM\n");
        
        int bit_depth = 0;
        if (maxval == 255) {
            bit_depth = 8;
        } else if (maxval == 65535) {
            bit_depth = 16;
        } else {
            imread_assert(false, "Invalid max bit-depth value in PPM\n");
        }
        
        imread_assert(std::strcmp(header, "P6") == 0 || std::strcmp(header, "p6") == 0,
                      "Input is not binary PPM\n");
        
        const int channels = 3;
        const int full_size = width * height * channels;
        std::unique_ptr<Image> im = factory->create(bit_depth, height, width, channels);
        
        /// convert the data to T
        if (bit_depth == 8) {
            uint8_t* __restrict__ data = new uint8_t[full_size];
            imread_assert(std::fread(static_cast<void*>(data),
                          sizeof(uint8_t), full_size, membuf.get()) == static_cast<std::size_t>(full_size),
                    "Could not read PPM 8-bit data\n");
            
            uint8_t* __restrict__ im_data = im->rowp_as<uint8_t>(0);
            for (int y = 0; y < height; y++) {
                uint8_t* __restrict__ row = static_cast<uint8_t*>(&data[(y*width)*channels]);
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        im_data[(c*height+y)*width+x] = *row++;
                    }
                }
            }
            delete[] data;
        } else if (bit_depth == 16) {
            uint16_t* __restrict__ data = new uint16_t[full_size];
            imread_assert(std::fread(static_cast<void*>(data),
                          sizeof(uint16_t), full_size, membuf.get()) == static_cast<std::size_t>(full_size),
                "Could not read PPM 16-bit data\n");
            
            uint16_t* __restrict__ im_data = im->rowp_as<uint16_t>(0);
            if (detail::littleendian()) {
                for (int y = 0; y < height; y++) {
                    uint16_t* __restrict__ row = static_cast<uint16_t*>(&data[(y*width)*channels]);
                    for (int x = 0; x < width; x++) {
                        uint16_t value;
                        for (int c = 0; c < channels; c++) {
                            value = *row++;
                            SWAP_ENDIAN16(value);
                            im_data[(c*height+y)*width+x] = value;
                        }
                    }
                }
            } else {
                for (int y = 0; y < height; y++) {
                    uint16_t* __restrict__ row = static_cast<uint16_t*>(&data[(y*width)*channels]);
                    for (int x = 0; x < width; x++) {
                        for (int c = 0; c < channels; c++) {
                            im_data[(c*height+y)*width+x] = *row++;
                        }
                    }
                }
            }
            delete[] data;
        }
        
        return im;
    }
    
    void PPMFormat::write(Image& input, byte_sink* output,
                          Options const& opts) {
        /// YO DOGG
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2); /// should be 3
        const int bit_depth = input.nbits();
        const int full_size = width * height * channels;
        
        /// write header
        output->writef("P6\n%d %d\n%d\n", width, height, (1<<bit_depth)-1);
        
        /// write data
        if (bit_depth == 8) {
            uint8_t* __restrict__ data = new uint8_t[full_size];
            av::strided_array_view<byte, 3> view = input.view();
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    uint8_t* __restrict__ p = static_cast<uint8_t*>(&data[(y*width+x)*channels]);
                    for (int c = 0; c < channels; c++) {
                        p[c] = view[{x, y, c}];
                    }
                }
            }
            imread_assert(output->write(data, full_size) == static_cast<std::size_t>(full_size),
                "Could not write 8-bit PPM data\n");
            delete[] data;
        } else if (bit_depth == 16) {
            av::strided_array_view<byte, 3> view = input.view();
            uint16_t* __restrict__ data = new uint16_t[full_size];
            if (detail::littleendian()) {
                uint16_t value;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        uint16_t* __restrict__ p = static_cast<uint16_t *>(&data[(y*width+x)*channels]);
                        for (int c = 0; c < channels; c++) {
                            value = view[{x, y, c}];
                            SWAP_ENDIAN16(value);
                            p[c] = value;
                        }
                    }
                }
            } else {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        uint16_t* __restrict__ p = static_cast<uint16_t *>(&data[(y*width+x)*channels]);
                        for (int c = 0; c < channels; c++) {
                            p[c] = view[{x, y, c}];
                        }
                    }
                }
            }
            imread_assert(output->write(data, full_size) == static_cast<std::size_t>(full_size),
                "Could not write 16-bit PPM data\n");
            delete[] data;
        }
        
        output->flush();
    }
}

