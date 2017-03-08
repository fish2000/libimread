/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <iostream>
#include <iod/json.hh>
#include <libimread/ext/base64.hh>
#include <libimread/IO/png.hh>
#include <libimread/pixels.hh>

#ifdef __APPLE__
    #include <libpng16/png.h>   /* this is the Homebrew path */
#else
    #include <png.h>            /* this is the standard location */
#endif

#define PP_CHECK(pp, msg) if (setjmp(png_jmpbuf(pp))) { imread_raise(CannotReadError, msg); }

namespace im {
    
    DECLARE_FORMAT_OPTIONS(PNGFormat);
    
    namespace {
        
        void throw_error(png_structp png_ptr, png_const_charp msg) {
            imread_raise(PNGIOError, msg);
        }
        
        class png_holder {
            public:
                png_holder(int m)
                    :png_ptr((m == write_mode ? png_create_write_struct : png_create_read_struct)(
                        PNG_LIBPNG_VER_STRING, 0, throw_error, 0)), png_info(0), mode(holder_mode(m))
                        {}
                
                ~png_holder() {
                    png_infopp pp = (png_info ? &png_info : 0);
                    if (mode == read_mode) {
                        png_destroy_read_struct(&png_ptr, pp, 0);
                    } else {
                        png_destroy_write_struct(&png_ptr, pp);
                    }
                }
                
                void create_info() {
                    png_info = png_create_info_struct(png_ptr);
                    if (!png_info) {
                        imread_raise(ProgrammingError,
                            "Error returned from png_create_info_struct");
                    }
                }
                
                png_structp png_ptr;
                png_infop png_info;
                enum holder_mode { read_mode, write_mode } mode;
        };
        
        void read_from_source(png_structp png_ptr, png_byte* buffer, std::size_t n) {
            byte_source* s = static_cast<byte_source*>(png_get_io_ptr(png_ptr));
            const std::size_t actual = s->read(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) { imread_raise_default(CannotReadError); }
        }
        
        void write_to_source(png_structp png_ptr, png_byte* buffer, std::size_t n) {
            byte_sink* s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
            const std::size_t actual = s->write(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) { imread_raise_default(CannotWriteError); }
        }
        
        void flush_source(png_structp png_ptr) {
            byte_sink* s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
            s->flush();
        }
        
        __attribute__((unused))
        int color_type_of(Image* im) {
            if (im->nbits() != 8 && im->nbits() != 16) {
                imread_raise(CannotWriteError,
                    "Image must be 8 or 16 bits for saving in PNG format");
            }
            if (im->ndims() == 2) { return PNG_COLOR_TYPE_GRAY; }
            if (im->ndims() != 3) {
                imread_raise(CannotWriteError,
                    "Image must be either 2 or 3 dimensional");
            }
            if (im->dim(2) == 3) { return PNG_COLOR_TYPE_RGB; }
            if (im->dim(2) == 4) { return PNG_COLOR_TYPE_RGBA; }
            imread_raise_default(CannotWriteError);
        }
        
        static png_byte color_types[4] = {
            PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA,
            PNG_COLOR_TYPE_RGB,  PNG_COLOR_TYPE_RGB_ALPHA
        };
        
        /// XXX: I kind of hate this thing, for not being
        /// a real allocator (which means I am an allocatorist, right?)
        struct stack_based_memory_pool {
            /// An allocator-ish RAII object,
            /// on stack exit it frees its allocations
            stack_based_memory_pool() { }
            
            ~stack_based_memory_pool() {
                for (unsigned i = 0; i != data.size(); ++i) {
                    operator delete(data[i]);
                    data[i] = 0;
                }
            }
            
            void* allocate(const int n) {
                data.reserve(data.size() + 1);
                void* d = operator new(n);
                data.push_back(d);
                return d;
            }
            
            template <typename T>
            T allocate_as(const int n) {
                return static_cast<T>(this->allocate(n));
            }
            
            private:
                std::vector<void*> data;
        };
        
        __attribute__((unused))
        void swap_bytes_inplace(std::vector<png_bytep>& data, const int ncols,
                                stack_based_memory_pool& mem) {
            for (unsigned int r = 0; r != data.size(); ++r) {
                png_bytep row = data[r];
                png_bytep newbf = mem.allocate_as<png_bytep>(ncols * 2);
                std::memcpy(newbf, row, ncols*2);
                for (int c = 0; c != ncols; ++c) {
                    std::swap(newbf[2*c], newbf[2*c + 1]);
                }
                data[r] = newbf;
            }
        }
        
        __attribute__((unused))
        int unknown_chunk_read_cb(png_structp ptr, png_unknown_chunkp chunk) {
            if (!strncmp(reinterpret_cast<char*>(chunk->name), "CgBI", 4)) {
                WTF("This file is already crushed and how the hell did you get here?");
                // std::exit(1);
            }
            return 1;
        }
        
        void swap_and_premultiply_alpha_transform(png_structp ptr,
                                                  png_row_infop row_info,
                                                  png_bytep data) {
            
            auto row_width = row_info->width;
            
            for (unsigned int x = 0; x < row_width * 4; x += 4) {
                png_byte r, g, b, a;
                r = data[x+0]; g = data[x+1]; b = data[x+2]; a = data[x+3];
                data[x+0] = (b*a) / 0xff;
                data[x+1] = (g*a) / 0xff;
                data[x+2] = (r*a) / 0xff;
            }
            
        }
        
    } /* namespace (anon.) */
    
    std::unique_ptr<Image> PNGFormat::read(byte_source* src, ImageFactory* factory, options_map const& opts) {
        png_holder p(png_holder::read_mode);
        png_set_read_fn(p.png_ptr, src, read_from_source);
        p.create_info();
        png_read_info(p.png_ptr, p.png_info);
        
        PP_CHECK(p.png_ptr, "PNG read struct setup failure");
        
        const int w =  png_get_image_width(p.png_ptr, p.png_info);
        const int h = png_get_image_height(p.png_ptr, p.png_info);
        volatile int channels = png_get_channels(p.png_ptr, p.png_info);
        volatile int bit_depth = png_get_bit_depth(p.png_ptr, p.png_info);
        
        if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
            imread_raise(CannotReadError,
                FF("Cannot read this bit depth ( %i ).", bit_depth),
                   "Only bit depths ∈ {1,8,16} are supported.");
        }
        if (bit_depth == 16 && !detail::bigendian()) { png_set_swap(p.png_ptr); }
        
        volatile int d = -1;
        const bool strip_alpha = opts.cast<bool>("png:strip_alpha", false);
        
        if (strip_alpha) {
            png_set_strip_alpha(p.png_ptr);
        }
        
        switch (png_get_color_type(p.png_ptr, p.png_info)) {
            case PNG_COLOR_TYPE_PALETTE:
                png_set_palette_to_rgb(p.png_ptr); /// ??
            case PNG_COLOR_TYPE_RGB:
                d = 3;
                break;
            case PNG_COLOR_TYPE_RGB_ALPHA:
                d = 4;
                break;
            case PNG_COLOR_TYPE_GRAY:
                //d = -1;
                d = 1;
                if (bit_depth < 8) {
                    png_set_expand_gray_1_2_4_to_8(p.png_ptr);
                }
                break;
            case PNG_COLOR_TYPE_GRAY_ALPHA:
                d = 1;
                imread_raise(CannotReadError,
                    "Color type ( 4: grayscale with alpha channel ) cannot be handled\n"
                    "without opts[\"png:strip_alpha\"] = true");
                break;
            default: {
                imread_raise(CannotReadError,
                    FF("Color type ( %i ) cannot be handled",
                        int(png_get_color_type(p.png_ptr, p.png_info))));
            }
        }
        
        // PP_CHECK(p.png_ptr, "PNG read elaboration failure");
        std::unique_ptr<Image> output(factory->create(bit_depth, h, w, d));
        
        /// GET METADATA (just ICC data for now)
        if (ImageWithMetadata* meta = dynamic_cast<ImageWithMetadata*>(output.get())) {
            if (png_get_valid(p.png_ptr, p.png_info, PNG_INFO_iCCP)) {
                /// extract the embedded ICC profile
                int compression;
                uint32_t length;
                byte* data;
                char* name;
                
                png_get_iCCP(p.png_ptr, p.png_info, &name,
                                                    &compression,
                                                    &data,
                                                    &length);
                
                meta->set_icc_name(std::string(name));
                meta->set_icc_data(&data[0], std::size_t(length));
            }
        }
        
        png_set_interlace_handling(p.png_ptr);
        png_read_update_info(p.png_ptr, p.png_info);
        
        volatile int row_bytes = png_get_rowbytes(p.png_ptr, p.png_info);
        png_bytep* __restrict__ row_pointers = new png_bytep[h];
        for (int y = 0; y < h; ++y) {
            row_pointers[y] = new png_byte[row_bytes];
        }
        
        // PP_CHECK(p.png_ptr, "PNG read rowbytes failure");
        
        png_read_image(p.png_ptr, row_pointers);
        
        PP_CHECK(p.png_ptr, "PNG read image failure");
        
        /// convert the data to T (fake it for now with uint8_t)
        //T *ptr = (T*)output->data();
        const int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t* __restrict__ ptr = static_cast<uint8_t*>(output->rowp_as<uint8_t>(0));
        
        // WTF("About to enter pixel access loop...",
        //     FF("w = %i, h = %i, channels = %i, bit_depth = %i, d = %i",
        //         w, h, channels, bit_depth, d));
        
        if (bit_depth == 8) {
            for (int y = 0; y < h; ++y) {
                uint8_t* __restrict__ srcPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (int x = 0; x < w; ++x) {
                    for (int c = 0; c < d; ++c) {
                        pix::convert(*srcPtr++, ptr[c*c_stride]);
                    }
                    ++ptr;
                }
                delete[] row_pointers[y];
            }
        } else if (bit_depth == 16) {
            for (int y = 0; y < h; ++y) {
                uint8_t* __restrict__ srcPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (int x = 0; x < w; ++x) {
                    for (int c = 0; c < d; ++c) {
                        uint16_t hi = (*srcPtr++) << 8;
                        uint16_t lo = hi | (*srcPtr++);
                        pix::convert(lo, ptr[c*c_stride]);
                    }
                    ++ptr;
                }
                delete[] row_pointers[y];
            }
        }
        
        /// clean up
        // for (int y = 0; y < h; y++) { delete[] row_pointers[y]; }
        delete[] row_pointers;
        
        return output;
    }
    
    void PNGFormat::write(Image& input, byte_sink* output, options_map const& opts) {
        png_holder p(png_holder::write_mode);
        p.create_info();
        png_set_write_fn(p.png_ptr, output, write_to_source, flush_source);
        
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2);
        const int bit_depth = input.nbits();
        
        png_bytep* __restrict__ row_pointers;
        const png_byte color_type = color_types[channels - 1];
        
        png_set_IHDR(p.png_ptr, p.png_info,
                     width, height, bit_depth, color_type,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                     PNG_FILTER_TYPE_BASE);
        
        const int compression = opts.cast<int>("png:compression", 0);
        if (compression) {
            png_set_compression_level(p.png_ptr, compression);
        }
        
        png_write_info(p.png_ptr, p.png_info);
        row_pointers = new png_bytep[height];
        
        const int c_stride = (channels == 1) ? 0 : input.stride(2);
        const int rowbytes = png_get_rowbytes(p.png_ptr, p.png_info);
        int x = 0, y = 0, c = 0;
        uint8_t* __restrict__ dstPtr;
        uint8_t out;
        
        if (bit_depth == 16) {
            // downconvert to uint8_t from uint16_t-ish data
            uint16_t* __restrict__ srcPtr = input.rowp_as<uint16_t>(0);
            
            for (y = 0; y < height; ++y) {
                row_pointers[y] = new png_byte[rowbytes];
                dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (x = 0; x < width; ++x) {
                    for (c = 0; c < channels; c++) {
                        pix::convert(srcPtr[c*c_stride], *dstPtr++);
                        // *dstPtr++ = out;
                    }
                    srcPtr++;
                }
            }
        } else if (bit_depth == 8) {
            // stick with uint8_t
            pix::accessor<byte> at = input.access();
            
            for (y = 0; y < height; ++y) {
                row_pointers[y] = new png_byte[rowbytes];
                dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (x = 0; x < width; ++x) {
                    for (c = 0; c < channels; c++) {
                        pix::convert(at(x, y, c)[0], *dstPtr++);
                    }
                }
            }
        } else {
            imread_assert(bit_depth == 8 || bit_depth == 16,
                "We only support saving 8- and 16-bit images.");
        }
        
        // write data
        // imread_assert(!setjmp(png_jmpbuf(p.png_ptr)),
        //     "[write_png_file] Error during writing bytes");
        png_write_image(p.png_ptr, row_pointers);
        
        // finish write
        imread_assert(!setjmp(png_jmpbuf(p.png_ptr)),
            "[write_png_file] Error during end of write");
        png_write_end(p.png_ptr, p.png_info);
        
        // clean up
        for (int y = 0; y < height; ++y) { delete[] row_pointers[y]; }
        delete[] row_pointers;
    }
    
    
    /// ADAPTED FROM PINCRUSH - FREAKY REFORMATTED PNG FOR IOS
    void PNGFormat::write_ios(Image& input, byte_sink* output, options_map const& opts) {
        /// immediately write the header, 
        /// before initializing the holder
        output->write(base64::decode(options.signatures[0].bytes).get(),
                                     options.signatures[0].length);
        
        png_holder p(png_holder::write_mode);
        p.create_info();
        png_set_write_fn(p.png_ptr, output, write_to_source, flush_source);
        png_set_sig_bytes(p.png_ptr, 8);
        
        /// not sure we need this
        png_set_filter(p.png_ptr, 0, PNG_FILTER_NONE);
        
        /// from the pincrush source:
        ///     "The default window size is 15 bits. Setting it to -15
        ///      causes zlib to discard the header and crc information.
        ///      This is critical to making a proper CgBI PNG"
        png_set_compression_window_bits(p.png_ptr, -15);
        
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2);
        const int bit_depth = input.nbits();
        
        png_bytep* __restrict__ row_pointers;
        png_byte color_type = color_types[channels - 1];
        
        if (!(color_type & PNG_COLOR_MASK_ALPHA)) {
            // Expand, adding an opaque alpha channel.
            FORSURE("[apple-png] Adding opaque alpha channel");
            png_set_add_alpha(p.png_ptr, 0xff, PNG_FILLER_AFTER);
        }
        
        png_set_read_user_transform_fn(p.png_ptr,
            swap_and_premultiply_alpha_transform);
        
        png_set_IHDR(p.png_ptr, p.png_info,
                     width, height, bit_depth, color_type,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                     PNG_FILTER_TYPE_BASE);
        
        // Standard Gamma
        png_set_gAMA(p.png_ptr, p.png_info, 0.45455);
        
        // Primary Chromaticities white_xy, red_xy, blue_xy, green_xy, in that order.
        png_set_cHRM(p.png_ptr, p.png_info,
            0.312700, 0.329000, 0.640000, 0.330000,
            0.300000, 0.600000, 0.150000, 0.060000);
        
        // Apple's PNGs have an sRGB intent of 0.
        png_set_sRGB(p.png_ptr, p.png_info, 0);
        
        /// This is the freaky custom Apple chunk
        static const png_byte cname[] = {  'C',  'g',  'B',  'I',  '\0' };
        static const png_byte cdata_solid[] = { 0x50, 0x00, 0x20, 0x06 };
        static const png_byte cdata_alpha[] = { 0x50, 0x00, 0x20, 0x02 };
        if (color_type & PNG_COLOR_MASK_ALPHA || color_type == PNG_COLOR_TYPE_PALETTE) {
            // I'm not sure, but if the input colortype is alpha anything, CgBI[3] is 0x02 instead of 0x06.
            // Strange, because 0x06 means RGBA and 0x02 does not.
            // But, Mimick this behaviour. Otherwise, our alpha channel is ignored.
            // cdata[3] = 0x02;
            png_write_chunk(p.png_ptr, cname, cdata_alpha, 4);
        } else {
            png_write_chunk(p.png_ptr, cname, cdata_solid, 4);
        }
        
        /// WRITE THE INFO STRUCT GAAAAH
        png_write_info(p.png_ptr, p.png_info);
        
        row_pointers = new png_bytep[height];
        
        /// NEED TO STOP HARDCODING IN UNSIGNED FUCKING CHAR FOR THE PIXEL TYPES SOMETIME
        int c_stride = (channels == 1) ? 0 : input.stride(2);
        int rowbytes = png_get_rowbytes(p.png_ptr, p.png_info);
        int x = 0, y = 0, c = 0;
        uint8_t* __restrict__ dstPtr;
        uint8_t out;
        
        if (bit_depth == 16) {
            // downconvert to uint8_t from uint16_t-ish data
            uint16_t* __restrict__ srcPtr = input.rowp_as<uint16_t>(0);
            
            for (y = 0; y < height; ++y) {
                row_pointers[y] = new png_byte[rowbytes];
                dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (x = 0; x < width; ++x) {
                    for (c = 0; c < channels; ++c) {
                        pix::convert(srcPtr[c*c_stride], out);
                        *dstPtr++ = out;
                    }
                    ++srcPtr;
                }
            }
        } else if (bit_depth == 8) {
            // stick with uint8_t
            uint8_t* __restrict__ srcPtr = input.rowp_as<uint8_t>(0);
            
            for (y = 0; y < height; ++y) {
                row_pointers[y] = new png_byte[rowbytes];
                dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (x = 0; x < width; ++x) {
                    for (c = 0; c < channels; ++c) {
                        pix::convert(srcPtr[c*c_stride], out);
                        *dstPtr++ = out;
                    }
                    ++srcPtr;
                }
            }
        } else {
            imread_assert(bit_depth == 8 || bit_depth == 16,
                "We only support saving 8- and 16-bit images.");
        }
        
        // write data
        imread_assert(!setjmp(png_jmpbuf(p.png_ptr)),
            "[write_png_file] Error during writing bytes");
        png_write_image(p.png_ptr, row_pointers);
        
        // finish write
        imread_assert(!setjmp(png_jmpbuf(p.png_ptr)),
            "[write_png_file] Error during end of write");
        png_write_end(p.png_ptr, p.png_info);
        
        // clean up
        for (int y = 0; y < height; ++y) { delete[] row_pointers[y]; }
        delete[] row_pointers;
    }
    
}
