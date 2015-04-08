// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <iostream>
#include <libimread/IO/png.hh>

#define PP_CHECK(pp, msg) if (setjmp(png_jmpbuf(pp))) { throw CannotReadError(msg); }

namespace im {
    
    namespace {
        
        void throw_error(png_structp png_ptr, png_const_charp msg) { throw CannotReadError(msg); }
        
        // This checks how 16-bit uints are stored in the current platform.
        inline bool is_big_endian() {
            uint16_t v = 0xff00;
            unsigned char *vp = reinterpret_cast<unsigned char*>(&v);
            return (*vp == 0xff);
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
                        throw ProgrammingError(
                            "_png.cpp: png_holder::create_info(): Error returned from png_create_info_struct"
                        ); 
                    }
                }
                
                png_structp png_ptr;
                png_infop png_info;
                enum holder_mode { read_mode, write_mode } mode;
        };
        
        void read_from_source(png_structp png_ptr, png_byte *buffer, size_t n) {
            byte_source *s = static_cast<byte_source*>(png_get_io_ptr(png_ptr));
            const size_t actual = s->read(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) {
                throw CannotReadError();
            }
        }
        
        void write_to_source(png_structp png_ptr, png_byte *buffer, size_t n) {
            byte_sink *s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
            const size_t actual = s->write(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) {
                throw CannotReadError();
            }
        }
        
        void flush_source(png_structp png_ptr) {
            byte_sink *s = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
            s->flush();
        }
        
        int color_type_of(Image *im) {
            if (im->nbits() != 8 && im->nbits() != 16) throw CannotWriteError(
                "_png.cpp: color_type_of(): Image must be 8 or 16 bits for saving in PNG format"
            );
            if (im->ndims() == 2) return PNG_COLOR_TYPE_GRAY;
            if (im->ndims() != 3) throw CannotWriteError(
                "_png.cpp: color_type_of(): Image must be either 2 or 3 dimensional"
            );
            if (im->dim(2) == 3) return PNG_COLOR_TYPE_RGB;
            if (im->dim(2) == 4) return PNG_COLOR_TYPE_RGBA;
            throw CannotWriteError();
        }
        
        void swap_bytes_inplace(std::vector<png_bytep> &data, const int ncols, stack_based_memory_pool &mem) {
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
    }
    
    std::unique_ptr<Image> PNGFormat::read(byte_source *src, ImageFactory *factory, const options_map &opts) {
        png_holder p(png_holder::read_mode);
        png_set_read_fn(p.png_ptr, src, read_from_source);
        p.create_info();
        png_read_info(p.png_ptr, p.png_info);
        
        PP_CHECK(p.png_ptr, "PNG read struct setup failure");
        
        const int w = png_get_image_width (p.png_ptr, p.png_info);
        const int h = png_get_image_height(p.png_ptr, p.png_info);
        int channels = png_get_channels(p.png_ptr, p.png_info);
        int bit_depth = png_get_bit_depth(p.png_ptr, p.png_info);
        
        if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
            std::ostringstream out;
            out << "im::PNGFormat::read(): Cannot read this bit depth ("
                    << bit_depth
                    << "). Only bit depths ∈ {1,8,16} are supported.";
            throw CannotReadError(out.str());
        }
        if (bit_depth == 16 && !is_big_endian()) { png_set_swap(p.png_ptr); }
        //if (bit_depth < 8) { png_set_packing(p.png_ptr); } /// ?
        
        int d = -1;
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
            default: {
                std::ostringstream out;
                out << "im::PNGFormat::read(): Color type ("
                    << int(png_get_color_type(p.png_ptr, p.png_info))
                    << ") cannot be handled";
                throw CannotReadError(out.str());
            }
        }
        
        PP_CHECK(p.png_ptr, "PNG read elaboration failure");
        
        png_set_interlace_handling(p.png_ptr);
        png_read_update_info(p.png_ptr, p.png_info);
        
        std::unique_ptr<Image> output(factory->create(bit_depth, h, w, d));
        
        int row_bytes = png_get_rowbytes(p.png_ptr, p.png_info);
        png_bytep *row_pointers = new png_bytep[h];
        for (int y = 0; y < h; y++) {
            row_pointers[y] = new png_byte[row_bytes];
        }
        
        PP_CHECK(p.png_ptr, "PNG read rowbytes failure");
        
        png_read_image(p.png_ptr, row_pointers);
        
        PP_CHECK(p.png_ptr, "PNG read image failure");
        
        /// convert the data to T (fake it for now with uint8_t)
        //T *ptr = (T*)output->data();
        int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t *ptr = static_cast<uint8_t*>(output->rowp_as<uint8_t>(0));
        
        if (bit_depth == 8) {
            for (int y = 0; y < h; y++) {
                uint8_t *srcPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (int x = 0; x < w; x++) {
                    for (int c = 0; c < d; c++) {
                        pix::convert(*srcPtr++, ptr[c*c_stride]);
                    }
                    ptr++;
                }
            }
        } else if (bit_depth == 16) {
            for (int y = 0; y < h; y++) {
                uint8_t *srcPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (int x = 0; x < w; x++) {
                    for (int c = 0; c < d; c++) {
                        uint16_t hi = (*srcPtr++) << 8;
                        uint16_t lo = hi | (*srcPtr++);
                        pix::convert(lo, ptr[c*c_stride]);
                    }
                    ptr++;
                }
            }
        }
        
        /// clean up
        for (int y = 0; y < h; y++) { delete[] row_pointers[y]; }
        delete[] row_pointers;
        
        return output;
    }

    void PNGFormat::write(Image &input, byte_sink *output, const options_map &opts) {
        png_holder p(png_holder::write_mode);
        stack_based_memory_pool alloc;
        //std::unique_ptr<Image> input_ptr(input);
        p.create_info();
        png_set_write_fn(p.png_ptr, output, write_to_source, flush_source);
        const int height = input.dim(0);
        const int width = input.dim(1);
        const int bit_depth = input.nbits();
        //const int color_type = color_type_of(input_ptr.get());
        const int color_type = PNG_COLOR_TYPE_RGB;
        
        png_set_IHDR(p.png_ptr, p.png_info, width, height,
                         bit_depth, color_type, PNG_INTERLACE_NONE,
                         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        int compression_level = get_optional_int(opts, "png:compression_level", -1);
        if (compression_level != -1) {
            png_set_compression_level(p.png_ptr, compression_level);
        }
        png_write_info(p.png_ptr, p.png_info);
        
        std::vector<png_bytep> rowps = input.allrows<png_byte>();
        if (bit_depth == 16 && !is_big_endian()) {
            swap_bytes_inplace(rowps, width, alloc);
        }
        
        png_write_image(p.png_ptr, &rowps[0]);
        png_write_end(p.png_ptr, p.png_info);
    }

}
