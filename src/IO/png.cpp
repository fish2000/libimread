/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <csetjmp>
#include <sstream>
#include <iostream>
#include <memory>
#include <array>

#include <iod/json.hh>

#include <libimread/endian.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/base64.hh>
#include <libimread/IO/png.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>

#include <zlib.h>

#ifdef __APPLE__
    #include <libpng16/png.h>   /* this is the Homebrew path */
#else
    #include <png.h>            /* this is the standard location */
#endif

#define PP_CHECK(pp, msg) if (setjmp(png_jmpbuf(pp))) { imread_raise(CannotReadError, msg); }

namespace im {
    
    DECLARE_FORMAT_OPTIONS(PNGFormat);
    
    namespace {
        
        class PNGByteArray {
            
            public:
                using value_type = png_byte;
                using pointer = png_bytep;
                using reference = std::add_lvalue_reference_t<value_type>;
                using const_reference = std::add_const_t<reference>;
                using size_type = std::size_t;
            
            public:
                /// Construct with height and “rowbytes” values:
                PNGByteArray(std::size_t h, std::size_t rb)
                    :data_ptr{ new png_bytep[h] }
                    ,m_height{ h }
                    ,m_rowbytes{ rb }
                    {
                        for (std::size_t idx = 0; idx < m_height; ++idx) {
                            data_ptr[idx] = new png_byte[m_rowbytes];
                        }
                    }
            
            public:
                /// Move constructor:
                PNGByteArray(PNGByteArray&& other) noexcept
                    :data_ptr(std::exchange(other.data_ptr, nullptr))
                    ,m_height(std::exchange(other.m_height, 0))
                    ,m_rowbytes(std::exchange(other.m_rowbytes, 0))
                    {
                        other.deallocate = false;
                    }
                
                /// Move assignment operator:
                PNGByteArray& operator=(PNGByteArray&& other) noexcept {
                    if (data_ptr != other.data_ptr) {
                        data_ptr = std::exchange(other.data_ptr, nullptr);
                        m_height = std::exchange(other.m_height, 0);
                        m_rowbytes = std::exchange(other.m_rowbytes, 0);
                        other.deallocate = false;
                    }
                    return *this;
                }
            
            public:
                /// Destructor:
                virtual ~PNGByteArray() {
                    if (deallocate) {
                        for (std::size_t idx = 0; idx < m_height; ++idx) {
                            delete[] data_ptr[idx];
                        }
                        delete[] data_ptr;
                    }
                }
            
            public:
                /// API - accessors:
                constexpr std::size_t height() const      { return m_height;              }
                constexpr std::size_t rowbytes() const    { return m_rowbytes;            }
                constexpr std::size_t size() const        { return m_height * m_rowbytes; }
                constexpr png_bytep*  data() const        { return data_ptr;              }
                constexpr operator png_bytep*() const     { return data_ptr;              }
            
            public:
                /// API - subscript operator:
                constexpr png_bytep operator[](std::size_t idx) const {
                    return data_ptr[idx];
                }
                
                /// API - “at(…)” a la std::vector<…> and friends:
                constexpr png_bytep at(std::size_t idx) const {
                    if (!(idx < m_height)) {
                        imread_raise(PNGIOError,
                            "PNGByteArray::at() called with out-of-bounds index",
                         FF("PNGByteArray::at( %u ) -> height = %u", idx,
                                                                     m_height));
                    }
                    return data_ptr[idx];
                }
                
                /// API - “at(…)” returns a specific byte, from a row and a byte index:
                constexpr png_byte at(std::size_t row_idx, std::size_t byte_idx) const {
                    if ((!(row_idx < m_height)) || (!(byte_idx < m_rowbytes))) {
                        imread_raise(PNGIOError,
                            "PNGByteArray::at() called with an out-of-bounds index",
                         FF("PNGByteArray::at( %u, %u ) -> height = %u, rowbytes = %u", row_idx,
                                                                                        byte_idx,
                                                                                        m_height,
                                                                                        m_rowbytes));
                    }
                    return data_ptr[row_idx][byte_idx];
                }
                
                template <typename CastType> inline
                constexpr CastType* at(std::size_t idx) const {
                    return reinterpret_cast<CastType*>(at(idx));
                }
                
                template <typename CastType> inline
                constexpr CastType at(std::size_t row_idx, std::size_t byte_idx) const {
                    return static_cast<CastType>(at(row_idx, byte_idx));
                }
            
            protected:
                png_bytep* __restrict__ data_ptr{ nullptr };    /// root data pointer
                std::size_t m_height{ 1 };                      /// “height” value
                std::size_t m_rowbytes{ 0 };                    /// “rowbytes” value
                bool deallocate = true;                         /// whether or not deallocation happens on destruction
            
            private:
                /// Disallow default-construct, copy-construct and copy-assign:
                constexpr PNGByteArray() noexcept = default;
                constexpr PNGByteArray(PNGByteArray const&);
                constexpr PNGByteArray& operator=(PNGByteArray const&);
            
        };
        
        
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
        
        static const std::array<int, 4> color_types{{
            PNG_COLOR_TYPE_GRAY,    PNG_COLOR_TYPE_GRAY_ALPHA,
            PNG_COLOR_TYPE_RGB,     PNG_COLOR_TYPE_RGB_ALPHA
        }};
        
        int color_type_for(std::size_t channel_count) {
            return color_types[channel_count - 1];
        }
        
        int channel_count_for(png_byte color_type) {
            auto found = std::find(std::begin(color_types),
                                   std::end(color_types),
                                   color_type);
            if (found == std::end(color_types)) { return 0; }
            return std::distance(std::begin(color_types), found) + 1;
        }
        
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
                virtual void finish() const = 0;
                virtual png_infop png_info() const = 0;
                
            public:
                int rowbytes() const {
                    return png_get_rowbytes(structure_ptr, png_info());
                }
                
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
                        png_set_read_fn(structure_ptr, source_ptr,
                                     [](png_structp struct_ptr, png_byte* buffer, std::size_t n) -> void {
                                          im::byte_source* source = static_cast<im::byte_source*>(png_get_io_ptr(struct_ptr));
                                          const std::size_t actual = source->read(reinterpret_cast<byte*>(buffer), n);
                                          if (actual != n) {
                                              imread_raise_default(CannotReadError);
                                          }
                                     });
                    }
            
            public:
                virtual ~PNGReader() {
                    png_infopp infop = info_ptr ? &info_ptr : nullptr;
                    png_destroy_read_struct(&structure_ptr, infop, 0);
                }
            
            public:
                virtual void start() const override {
                    png_read_info(structure_ptr, info_ptr);
                    if (png_get_bit_depth(structure_ptr, info_ptr) == 16 && !detail::bigendian()) {
                        PNGErrorReporter::set_swap();
                    }
                }
                
            public:
                virtual void update() const override {
                    png_read_update_info(structure_ptr, info_ptr);
                }
                
            public:
                virtual void finish() const override {}
                
            public:
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
                
            public:
                bool has_valid_ICC_data() const {
                    return png_get_valid(structure_ptr, info_ptr, PNG_INFO_iCCP);
                }
                
            public:
                PNGByteArray bytearray() const {
                    PNGByteArray out(height(), rowbytes());
                    png_read_image(structure_ptr, out.data());
                    return out;
                }
                
            protected:
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
                    }
            
            public:
                ~PNGWriter() {
                    png_infopp infop = info_ptr ? &info_ptr : nullptr;
                    png_destroy_write_struct(&structure_ptr, infop);
                }
            
            public:
                virtual void start() const override {
                    png_write_info(structure_ptr, info_ptr);
                }
                
            public:
                int width() const               { return dimensions.width;                              }
                int height() const              { return dimensions.height;                             }
                int channels() const            { return dimensions.channels;                           }
                int bit_depth() const           { return dimensions.bit_depth;                          }
                png_byte color_type() const     { return dimensions.color_type;                         }
                png_byte interlace() const      { return dimensions.interlace;                          }
                png_byte compression() const    { return dimensions.compression;                        }
                png_byte filter() const         { return dimensions.filter;                             }
                
            public:
                void set_width(int width)       { dimensions.width = width;                             }
                void set_height(int height)     { dimensions.height = height;                           }
                void set_channels(int channels) { dimensions.channels = channels;
                                                  dimensions.color_type = color_type_for(channels);     }
                void set_bit_depth(int depth)   { dimensions.bit_depth = depth;                         }
                void set_color_type(int coltyp) { dimensions.color_type = coltyp;
                                                  dimensions.channels = channel_count_for(coltyp);      }
                
                void set_compression(int level) { png_set_compression_level(structure_ptr, level); }
                
            public:
                void retrieve_headers() const   { dimensions.from_IHDR(structure_ptr, info_ptr); } /// populates “dimension”
                void commit_headers() const     { dimensions.to_IHDR(structure_ptr, info_ptr); } /// writes “dimension” values
                
                virtual void update() const override { dimensions.to_IHDR(structure_ptr, info_ptr); }
                
            public:
                PNGByteArray bytearray() const {
                    return PNGByteArray(height(), rowbytes());
                }
                
                void write_bytes(png_bytep* row_pointers) const {
                    if (row_pointers) {
                        png_write_image(structure_ptr, row_pointers);
                    }
                }
                
            public:
                virtual void finish() const override {
                    png_write_end(structure_ptr, info_ptr);
                }
                
            public:
                virtual png_infop png_info() const override {
                    return info_ptr;
                }
                
            protected:
                mutable byte_sink* sink_ptr;
                mutable png_infop info_ptr;
                mutable struct dimension_cache_t {
                    uint32_t width{0},
                             height{0},
                             channels{0};
                     int32_t bit_depth{0},
                             color_type{PNG_COLOR_TYPE_RGB},
                             interlace{PNG_INTERLACE_NONE},
                             compression{PNG_COMPRESSION_TYPE_BASE},
                             filter{PNG_FILTER_TYPE_BASE};
                    void to_IHDR(png_structp structure, png_infop info) {
                                       png_set_IHDR(structure, info,
                                                    width, height,
                                                    bit_depth, color_type,
                                                    interlace, compression, filter); }
                    void from_IHDR(png_structp structure, png_infop info) {
                                       png_get_IHDR(structure, info,
                                                    &width, &height,
                                                    &bit_depth, &color_type,
                                                    &interlace, &compression, &filter);
                                       channels = channel_count_for(color_type); }
                } dimensions;
        
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
        /// Create our PNGReader wrapper-class instance:
        PNGReader reader(src);
        
        /// Start the decompression immediately:
        reader.start();
        
        /// error check: initial
        if (reader.has_error()) {
            imread_raise(PNGIOError,
                "PNG read struct setup failure");
        }
        
        /// store the width/height/channels/bit depth image information:
        const int w = reader.width();
        const int h = reader.height();
        const int channels = reader.channels();
        const int bit_depth = reader.bit_depth();
        
        /// error-check the bit depth we read from the PNG structure:
        if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
            imread_raise(CannotReadError,
                FF("Cannot read this bit depth ( %i ).", bit_depth),
                   "Only bit depths ∈ {1,8,16} are supported.");
        }
        
        /// set “swap” if we are big-endian and the bit depth is high enough:
        // if (bit_depth == 16 && !detail::bigendian()) { p.set_swap(); }
        
        // volatile int d = -1;
        const bool strip_alpha = opts.cast<bool>("png:strip_alpha", false);
        
        /// Strip alpha channel, if we asked to do so:
        if (strip_alpha) {
            reader.set_strip_alpha();
        }
        
        /// figure out the (potentially problematic) PNG color data situation:
        const int d = reader.components();
        
        /// error check: after “elaboration”
        if (reader.has_error()) {
            imread_raise(PNGIOError,
                "PNG read elaboration failure");
        }
        
        /// create the output image:
        std::unique_ptr<Image> output(factory->create(bit_depth, h, w, d));
        
        /// GET METADATA - just ICC data for now
        if (Metadata* meta = dynamic_cast<Metadata*>(output.get())) {
            if (reader.has_valid_ICC_data()) {
                /// extract the embedded ICC profile
                int compression;
                uint32_t length;
                byte* data;
                char* name;
                
                png_get_iCCP(reader.png_ptr(), reader.png_info(), &name,
                                                                  &compression,
                                                                  &data,
                                                                  &length);
                
                meta->set_icc_name(std::string(name));
                meta->set_icc_data(&data[0], std::size_t(length));
            }
        }
        
        /// set interlace handling:
        reader.set_interlace_handling();
        
        /// UPDATE the info struct --
        /// note no analog to this in PNGFormat::write():
        reader.update();
        
        /// allocate and fill row pointer byte array:
        PNGByteArray bytes(reader.bytearray());
        
        /// error check: after row pointer allocation and preparation:
        if (reader.has_error()) {
            imread_raise(PNGIOError,
                "PNG read/allocate rowbytes failure");
        }
        
        /// convert the data to T (fake it for now with uint8_t)
        const int c_stride = (d == 1) ? 0 : output->stride(2);
        uint8_t* __restrict__ ptr = output->rowp_as<uint8_t>(0);
        
        // WTF("About to enter pixel access loop...",
        //     FF("w = %i, h = %i, channels = %i, bit_depth = %i, d = %i",
        //         w, h, channels, bit_depth, d));
        
        if (bit_depth == 8) {
            int x{0}, y{0}, c{0}, cc{0};
            for (; y < h; ++y) {
                uint8_t* __restrict__ srcPtr = bytes.at<uint8_t>(y);
                for (; x < w; ++x) {
                    for (c = cc = 0; c < d; ++c, cc += c_stride) {
                        ptr[cc] = *srcPtr++;
                    }
                    ++ptr;
                }
            }
        } else if (bit_depth == 16) {
            int x{0}, y{0},
                c{0}, cc{0};
             uint16_t hi{0},
                      lo{0};
            for (; y < h; ++y) {
                uint8_t* __restrict__ srcPtr = bytes.at<uint8_t>(y);
                for (; x < w; ++x) {
                    for (c = cc = 0; c < d; ++c, cc += c_stride) {
                        hi = (*srcPtr++) << 8;
                        lo = hi | (*srcPtr++);
                        ptr[cc] = lo;
                    }
                    ++ptr;
                }
            }
        }
        
        /// finish up what needs finishing:
        reader.finish();
        
        /// return newly created image:
        return output;
    }
    
    void PNGFormat::write(Image& input,
                          byte_sink* output,
                          Options const& opts) {
        /// create our PNGWriter wrapper-class instance:
        PNGWriter writer(output);
        
        /// store the width/height/channels/bit depth image info:
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2);
        const int bit_depth = input.nbits();
        
        /// error-check the bit depth we read from the Image instance:
        if (bit_depth != 1 && bit_depth != 8 && bit_depth != 16) {
            imread_raise(CannotWriteError,
                FF("Cannot write this bit depth ( %i ).", bit_depth),
                   "Only bit depths ∈ {1,8,16} are supported.");
        }
        
        // png_bytep* __restrict__ row_pointers;
        
        /// map the channel count to a PNG color constant --
        /// q.v. the static array allocated in the anon. namespace above:
        // const png_byte color_type = color_type_as_channel_count(channels);
        
        /// set the dimensions and other meta-info values:
        writer.set_width(width);
        writer.set_height(height);
        writer.set_channels(channels);
        writer.set_bit_depth(bit_depth);
        
        /// set the compression level, as we specified:
        // const int zlevel = opts.cast<int>("png:compression", Z_BEST_COMPRESSION);
        const int zlevel = opts.cast<int>("png:compression", 0);
        if (zlevel) {
            // png_set_compression_level(writer.png_ptr(), compression);
            writer.set_compression(zlevel);
        }
        
        /// update libpng’s PNG struct with the latest dimensions:
        writer.update();
        
        // png_set_IHDR(writer.png_ptr(),
        //              writer.png_info(),
        //              width, height, bit_depth,
        //              color_type_for(channels),
        //              PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
        //                                  PNG_FILTER_TYPE_BASE);
        
        /// start the compression!
        writer.start();
        
        /// error check: initial
        if (writer.has_error()) {
            imread_raise(PNGIOError,
                "PNG write struct setup failure");
        }
        
        /// allocate byte array for row_pointers per the image height:
        // row_pointers = new png_bytep[height];
        
        PNGByteArray bytes(writer.bytearray());
        
        /// sort out the stride and rowbytes situation:
        const int c_stride = (channels == 1) ? 0 : input.stride(2);
        // const int rowbytes = png_get_rowbytes(writer.png_ptr(), writer.png_info());
        int x{0}, y{0}, c{0}, cc{0};
        uint8_t* __restrict__ dstPtr;
        uint8_t out;
        
        if (bit_depth == 16) {
            // downconvert to uint8_t from uint16_t-ish data
            uint16_t* __restrict__ srcPtr = input.rowp_as<uint16_t>(0);
            
            for (; y < height; ++y) {
                // row_pointers[y] = new png_byte[rowbytes];
                // dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                dstPtr = bytes.at<uint8_t>(y);
                for (; x < width; ++x) {
                    for (; c < channels; ++c) {
                        *dstPtr++ = srcPtr[c*c_stride] >> 8;
                    }
                    srcPtr++;
                }
            }
        } else if (bit_depth == 8) {
            // stick with uint8_t
            av::strided_array_view<uint8_t, 3> view = input.view<uint8_t>();
            
            for (; y < height; ++y) {
                // row_pointers[y] = new png_byte[rowbytes];
                // dstPtr = static_cast<uint8_t*>(row_pointers[y]);
                dstPtr = bytes.at<uint8_t>(y);
                for (; x < width; ++x) {
                    // av::strided_array_view<uint8_t, 1> pixel = view[x][y];
                    for (; c < channels; c++) {
                        *dstPtr++ = view[{ x, y, c }];
                        // *dstPtr++ = pixel[c];
                    }
                }
            }
        } else {
            imread_assert(bit_depth == 8 || bit_depth == 16,
                "We only support saving 8- and 16-bit images.");
        }
        
        // write data
        // imread_assert(!setjmp(png_jmpbuf(writer.png_ptr)),
        //     "[write_png_file] Error during writing bytes");
        // png_write_image(writer.png_ptr(), row_pointers);
        writer.write_bytes(bytes.data());
        
        // finish write
        // imread_assert(!setjmp(png_jmpbuf(writer.png_ptr())),
        //     "[write_png_file] Error during end of write");
        // png_write_end(writer.png_ptr(), writer.png_info());
        writer.finish();
        
        /// error check: final
        if (writer.has_error()) {
            imread_raise(PNGIOError,
                "PNG error after calling PNGWriter::finish()");
        }
        
        // clean up
        // for (int y = 0; y < height; ++y) { delete[] row_pointers[y]; }
        // delete[] row_pointers;
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
