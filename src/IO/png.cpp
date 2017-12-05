/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <csetjmp>
#include <sstream>
#include <iostream>

#include <iod/json.hh>

#include <libimread/endian.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/base64.hh>
#include <libimread/IO/png.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>

#ifdef __APPLE__
    #include <libpng16/png.h>   /* this is the Homebrew path */
#else
    #include <png.h>            /* this is the standard location */
#endif

#define PP_CHECK(pp, msg) if (setjmp(png_jmpbuf(pp))) { imread_raise(CannotReadError, msg); }

namespace im {
    
    DECLARE_FORMAT_OPTIONS(PNGFormat);
    
    namespace {
        
        class png_holder {
            public:
                png_holder(int m)
                    :png_ptr((m == write_mode ? png_create_write_struct
                                              : png_create_read_struct)(
                        PNG_LIBPNG_VER_STRING, 0,
                        [](png_structp png_ptr, png_const_charp message) -> void {
                            imread_raise(PNGIOError, message);
                        }, 0))
                    ,png_info(png_create_info_struct(png_ptr))
                    ,mode(static_cast<holder_mode>(m))
                    {
                        if (!png_info) {
                            imread_raise(ProgrammingError,
                                "Error returned from png_create_info_struct");
                        }
                    }
                
            public:
                ~png_holder() {
                    png_infopp pp = (png_info ? &png_info : nullptr);
                    if (mode == read_mode) {
                        png_destroy_read_struct(&png_ptr, pp, 0);
                    } else {
                        png_destroy_write_struct(&png_ptr, pp);
                    }
                }
                
            public:
                void start(im::seekable* source_sink) {
                    if (mode == read_mode) {
                        png_set_read_fn(png_ptr,
                                        dynamic_cast<im::byte_source*>(source_sink),
                                        [](png_structp png_ptr, png_byte* buffer, std::size_t n) -> void {
                                            im::byte_source* source = static_cast<im::byte_source*>(
                                                                      png_get_io_ptr(png_ptr));
                                            const std::size_t actual = source->read(reinterpret_cast<byte*>(buffer), n);
                                            if (actual != n) {
                                                imread_raise_default(CannotReadError);
                                            }
                                        });
                        png_read_info(png_ptr, png_info);
                    } else {
                        png_set_write_fn(png_ptr,
                                         dynamic_cast<im::byte_sink*>(source_sink),
                                         [](png_structp png_ptr, png_byte* buffer, std::size_t n) -> void {
                                             im::byte_sink* sink = static_cast<im::byte_sink*>(
                                                                   png_get_io_ptr(png_ptr));
                                             const std::size_t actual = sink->write(reinterpret_cast<byte*>(buffer), n);
                                             if (actual != n) {
                                                 imread_raise_default(CannotWriteError);
                                             }
                                         },
                                         [](png_structp png_ptr) -> void {
                                             im::byte_sink* sink = static_cast<im::byte_sink*>(png_get_io_ptr(png_ptr));
                                             sink->flush();
                                         });
                        png_write_info(png_ptr, png_info);
                    }
                }
                
            public:
                png_structp png_ptr;
                png_infop png_info;
                
            public:
                enum holder_mode { read_mode, write_mode } mode;
        };
        
        class PNGErrorReporter {
            
            public:
                PNGErrorReporter(png_structp struct_ptr, bool m)
                    :structure_ptr(struct_ptr)
                    ,mode(static_cast<reporter_mode>(m))
                    {}
            
            public:
                virtual ~PNGErrorReporter() {}
            
            public:
                bool has_error() const {
                    if (structure_ptr) {
                        return setjmp(png_jmpbuf(structure_ptr));
                    }
                    return true;
                }
                
                void set_swap() const {
                    png_set_swap(structure_ptr);
                }
                
                void set_strip_alpha() const {
                    png_set_strip_alpha(structure_ptr);
                }
                
                void set_interlace_handling() const {
                    png_set_interlace_handling(structure_ptr);
                }
                
            public:
                png_structp png_ptr() const {
                    return structure_ptr;
                }
                
            public:
                virtual void start() const = 0;
                virtual void update() const = 0;
                virtual png_infop png_info() const = 0;
                
            public:
                mutable png_structp structure_ptr;
                
            public:
                enum reporter_mode : bool {  read_mode = false,
                                            write_mode = true  } mode;
            
        };
        
        class PNGReader : public PNGErrorReporter {
            
            public:
                PNGReader(byte_source* source)
                    :PNGErrorReporter(png_create_read_struct(
                        PNG_LIBPNG_VER_STRING, 0,
                        [](png_structp struct_ptr, png_const_charp message) -> void {
                            imread_raise(PNGIOError, message);
                        }, 0), false)
                    ,source_ptr(source)
                    ,info_ptr(png_create_info_struct(structure_ptr))
                    {
                        if (!structure_ptr) {
                            imread_raise(ProgrammingError,
                                "Error returned from png_create_read_struct");
                        }
                        if (!info_ptr) {
                            imread_raise(ProgrammingError,
                                "Error returned from png_create_info_struct");
                        }
                    }
            
            public:
                virtual ~PNGReader() {
                    png_infopp infop = info_ptr ? &info_ptr : nullptr;
                    png_destroy_read_struct(&structure_ptr, infop, 0);
                }
            
            public:
                virtual void start() const override {
                    png_set_read_fn(structure_ptr, source_ptr,
                                 [](png_structp struct_ptr, png_byte* buffer, std::size_t n) -> void {
                                      im::byte_source* source = static_cast<im::byte_source*>(png_get_io_ptr(struct_ptr));
                                      const std::size_t actual = source->read(reinterpret_cast<byte*>(buffer), n);
                                      if (actual != n) {
                                          imread_raise_default(CannotReadError);
                                      }
                                 });
                    png_read_info(structure_ptr, info_ptr);
                    if (png_get_bit_depth(structure_ptr, info_ptr) == 16 && !detail::bigendian()) {
                        PNGErrorReporter::set_swap();
                    }
                }
                
                virtual void update() const override {
                    png_read_update_info(structure_ptr, info_ptr);
                }
                
                virtual png_infop png_info() const override {
                    return info_ptr;
                }
                
            public:
                int width() const           { return png_get_image_width(structure_ptr, info_ptr);  }
                int height() const          { return png_get_image_height(structure_ptr, info_ptr); }
                int channels() const        { return png_get_channels(structure_ptr, info_ptr);     }
                int bit_depth() const       { return png_get_bit_depth(structure_ptr, info_ptr);    }
                
                int components() const {
                    if (image_components == 0) {
                        switch (png_get_color_type(structure_ptr, info_ptr)) {
                            case PNG_COLOR_TYPE_PALETTE:
                                png_set_palette_to_rgb(structure_ptr); /// ??
                            case PNG_COLOR_TYPE_RGB:
                                image_components = 3;
                                break;
                            case PNG_COLOR_TYPE_RGB_ALPHA:
                                image_components = 4;
                                break;
                            case PNG_COLOR_TYPE_GRAY:
                                image_components = 1;
                                if (png_get_bit_depth(structure_ptr, info_ptr) < 8) {
                                    png_set_expand_gray_1_2_4_to_8(structure_ptr);
                                }
                                break;
                            case PNG_COLOR_TYPE_GRAY_ALPHA:
                                image_components = 1;
                                imread_raise(PNGIOError,
                                    "Color type ( 4: grayscale with alpha channel ) cannot be handled\n"
                                    "without opts[\"png:strip_alpha\"] = true");
                                break;
                            default: {
                                imread_raise(PNGIOError,
                                    FF("Color type ( %i ) cannot be handled",
                                        int(png_get_color_type(structure_ptr, info_ptr))));
                            }
                        }
                    }
                    return image_components;
                }
                
                bool has_valid_ICC_data() const {
                    return png_get_valid(structure_ptr, info_ptr, PNG_INFO_iCCP);
                }
                
            public:
                mutable byte_source* source_ptr;
                mutable png_infop info_ptr;
                mutable int image_components{ 0 };
        
        };
        
        class PNGWriter : public PNGErrorReporter {
            
            public:
                PNGWriter(byte_sink* sink)
                    :PNGErrorReporter(png_create_write_struct(
                        PNG_LIBPNG_VER_STRING, 0,
                        [](png_structp struct_ptr, png_const_charp message) -> void {
                            imread_raise(PNGIOError, message);
                        }, 0), true)
                    ,sink_ptr(sink)
                    ,info_ptr(png_create_info_struct(structure_ptr))
                    {
                        if (!structure_ptr) {
                            imread_raise(ProgrammingError,
                                "Error returned from png_create_write_struct");
                        }
                        if (!info_ptr) {
                            imread_raise(ProgrammingError,
                                "Error returned from png_create_info_struct");
                        }
                    }
            
            public:
                ~PNGWriter() {
                    png_infopp infop = info_ptr ? &info_ptr : nullptr;
                    png_destroy_write_struct(&structure_ptr, infop);
                }
            
            public:
                virtual void start() const override {
                    png_set_write_fn(structure_ptr, sink_ptr,
                                  [](png_structp struct_ptr, png_byte* buffer, std::size_t n) -> void {
                                      im::byte_sink* sink = static_cast<im::byte_sink*>(png_get_io_ptr(struct_ptr));
                                      const std::size_t actual = sink->write(reinterpret_cast<byte*>(buffer), n);
                                      if (actual != n) {
                                          imread_raise_default(CannotWriteError);
                                      }
                                  },
                                  [](png_structp struct_ptr) -> void {
                                      static_cast<im::byte_sink*>(png_get_io_ptr(struct_ptr))->flush();
                                  });
                    png_write_info(structure_ptr, info_ptr);
                }
                
                virtual void update() const override {}
                
                virtual png_infop png_info() const override {
                    return info_ptr;
                }
                
            public:
                mutable byte_sink* sink_ptr;
                mutable png_infop info_ptr;
        
        };
        
        void read_from_source(png_structp png_ptr, png_byte* buffer, std::size_t n) {
            byte_source* source = static_cast<byte_source*>(png_get_io_ptr(png_ptr));
            const std::size_t actual = source->read(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) {
                imread_raise_default(CannotReadError);
            }
        }
        
        void write_to_source(png_structp png_ptr, png_byte* buffer, std::size_t n) {
            byte_sink* sink = static_cast<byte_sink*>(png_get_io_ptr(png_ptr));
            const std::size_t actual = sink->write(reinterpret_cast<byte*>(buffer), n);
            if (actual != n) {
                imread_raise_default(CannotWriteError);
            }
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
            if (!std::strncmp(reinterpret_cast<char*>(chunk->name), "CgBI", 4)) {
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
    
    std::unique_ptr<Image> PNGFormat::read(byte_source* src,
                                           ImageFactory* factory,
                                           Options const& opts) {
        PNGReader p(src);
        p.start();
        
        /// error check: initial
        if (p.has_error()) {
            imread_raise(PNGIOError,
                "PNG read struct setup failure");
        }
        
        /// store the width/height/channels/bit depth image information:
        const int w = p.width();
        const int h = p.height();
        const int channels = p.channels();
        const int bit_depth = p.bit_depth();
        
        /// error-check the bit depth we read from the PNG structure:
        if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
            imread_raise(CannotReadError,
                FF("Cannot read this bit depth ( %i ).", bit_depth),
                   "Only bit depths ∈ {1,8,16} are supported.");
        }
        
        /// set “swap” if we are big-endian and the bit depth is high enough:
        if (bit_depth == 16 && !detail::bigendian()) { p.set_swap(); }
        
        // volatile int d = -1;
        const bool strip_alpha = opts.cast<bool>("png:strip_alpha", false);
        
        /// Strip alpha channel, if we asked to do so:
        if (strip_alpha) {
            p.set_strip_alpha();
        }
        
        /// figure out the (potentially problematic) PNG color data situation:
        const int d = p.components();
        
        /// error check: after “elaboration”
        if (p.has_error()) {
            imread_raise(PNGIOError,
                "PNG read elaboration failure");
        }
        
        /// create the output image:
        std::unique_ptr<Image> output(factory->create(bit_depth, h, w, d));
        
        /// GET METADATA - just ICC data for now
        if (Metadata* meta = dynamic_cast<Metadata*>(output.get())) {
            if (p.has_valid_ICC_data()) {
                /// extract the embedded ICC profile
                int compression;
                uint32_t length;
                byte* data;
                char* name;
                
                png_get_iCCP(p.png_ptr(), p.png_info(), &name,
                                                        &compression,
                                                        &data,
                                                        &length);
                
                meta->set_icc_name(std::string(name));
                meta->set_icc_data(&data[0], std::size_t(length));
            }
        }
        
        /// set interlace handling:
        // png_set_interlace_handling(p.png_ptr());
        p.set_interlace_handling();
        
        /// UPDATE the info struct --
        /// note no analog to this in PNGFormat::write():
        // png_read_update_info(p.png_ptr(), p.png_info());
        p.update();
        
        /// allocate:
        const int row_bytes = png_get_rowbytes(p.png_ptr(), p.png_info());
        png_bytep* __restrict__ row_pointers = new png_bytep[h];
        for (int y = 0; y < h; ++y) {
            row_pointers[y] = new png_byte[row_bytes];
        }
        
        /// error check: after rowbytes allocation
        if (p.has_error()) {
            imread_raise(PNGIOError,
                "PNG read/allocate rowbytes failure");
        }
        
        /// actually read the image data into row_pointers:
        png_read_image(p.png_ptr(), row_pointers);
        
        /// error check: after image data read
        if (p.has_error()) {
            imread_raise(PNGIOError,
                "PNG read image failure");
        }
        
        
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
                        ptr[c*c_stride] = *srcPtr++;
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
                        ptr[c*c_stride] = lo;
                    }
                    ++ptr;
                }
                delete[] row_pointers[y];
            }
        }
        
        /// clean up
        delete[] row_pointers;
        
        return output;
    }
    
    void PNGFormat::write(Image& input,
                          byte_sink* output,
                          Options const& opts) {
        /// create outr PNGWriter wrapper-class instance:
        PNGWriter p(output);
        
        /// store the width/height/channels/bit depth image info:
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2);
        const int bit_depth = input.nbits();
        
        png_bytep* __restrict__ row_pointers;
        
        /// map the channel count to a PNG color constant --
        /// q.v. the static array allocated in the anon. namespace above:
        const png_byte color_type = color_types[channels - 1];
        
        png_set_IHDR(p.png_ptr(), p.png_info(),
                     width, height, bit_depth, color_type,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                     PNG_FILTER_TYPE_BASE);
        
        /// set the compression level, as we specified:
        const int compression = opts.cast<int>("png:compression", 0);
        if (compression) {
            png_set_compression_level(p.png_ptr(), compression);
        }
        
        /// start the compression!
        p.start();
        
        /// allocate byte array for row_pointers per the image height:
        row_pointers = new png_bytep[height];
        
        /// sort out the stride and rowbytes situation:
        const int c_stride = (channels == 1) ? 0 : input.stride(2);
        const int rowbytes = png_get_rowbytes(p.png_ptr(), p.png_info());
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
                        *dstPtr++ = srcPtr[c*c_stride];
                    }
                    srcPtr++;
                }
            }
        } else if (bit_depth == 8) {
            // stick with uint8_t
            av::strided_array_view<byte, 3> view = input.view();
            
            for (y = 0; y < height; ++y) {
                row_pointers[y] = new png_byte[rowbytes];
                dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                for (x = 0; x < width; ++x) {
                    for (c = 0; c < channels; c++) {
                        *dstPtr++ = view[{x, y, c}];
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
        png_write_image(p.png_ptr(), row_pointers);
        
        // finish write
        imread_assert(!setjmp(png_jmpbuf(p.png_ptr())),
            "[write_png_file] Error during end of write");
        png_write_end(p.png_ptr(), p.png_info());
        
        // clean up
        for (int y = 0; y < height; ++y) { delete[] row_pointers[y]; }
        delete[] row_pointers;
    }
    
    
    /// ADAPTED FROM PINCRUSH - FREAKY REFORMATTED PNG FOR IOS
    void PNGFormat::write_ios(Image& input, byte_sink* output, Options const& opts) {
        /// immediately write the header, 
        /// before initializing the holder
        output->write(base64::decode(options.signatures[0].bytes).get(),
                                     options.signatures[0].length);
        
        png_holder p(png_holder::write_mode);
        // p.create_info();
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
        png_set_gAMA(p.png_ptr, p.png_info,
                     options.standard_gamma);
        
        // Primary Chromaticities white_xy, red_xy, blue_xy, green_xy, in that order.
        png_set_cHRM(p.png_ptr, p.png_info,
                     options.primary_cromacities.white.xx,
                     options.primary_cromacities.white.yy,
                     options.primary_cromacities.red.xx,
                     options.primary_cromacities.red.yy,
                     options.primary_cromacities.blue.xx,
                     options.primary_cromacities.blue.yy,
                     options.primary_cromacities.green.xx,
                     options.primary_cromacities.green.yy);
        
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
                        out = srcPtr[c*c_stride];
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
                        out = srcPtr[c*c_stride];
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
